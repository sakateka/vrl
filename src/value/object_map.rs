//! An ordered object map with exact accounting of its allocated node storage.

use std::{
    borrow::Borrow,
    cell::Cell,
    collections::BTreeMap as StdMap,
    fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    ops::{Index, RangeBounds},
    ptr::NonNull,
};

use arena_btreemap::{
    BTreeMap,
    alloc::{AllocError, Allocator, Global, Layout},
    btree::map,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{KeyString, Value};

#[repr(align(64))]
struct ThreadCounter(Cell<*mut usize>);

thread_local! {
    static ACTIVE_COUNTER: ThreadCounter = const { ThreadCounter(Cell::new(std::ptr::null_mut())) };
}

#[inline]
fn with_counter<T>(counter: &mut usize, operation: impl FnOnce() -> T) -> T {
    ACTIVE_COUNTER.with(|context| {
        struct Restore<'a>(&'a Cell<*mut usize>, *mut usize);
        impl Drop for Restore<'_> {
            #[inline]
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _restore = Restore(&context.0, context.0.replace(std::ptr::from_mut(counter)));
        operation()
    })
}

#[derive(Clone, Copy, Default)]
struct NodeAllocator;

// SAFETY: Blocks and their original layouts are delegated to Global. Every allocating or
// deallocating map operation installs its own live counter for that synchronous call only.
unsafe impl Allocator for NodeAllocator {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        ACTIVE_COUNTER.with(|context| {
            let counter = context.0.get();
            if counter.is_null() {
                return Err(AllocError);
            }
            let block = Global.allocate(layout)?;
            // SAFETY: with_counter exclusively borrows the owning map/iterator's counter.
            unsafe {
                counter.write(
                    counter
                        .read()
                        .checked_add(layout.size())
                        .expect("node byte overflow"),
                );
            }
            Ok(block)
        })
    }
    #[inline]
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        ACTIVE_COUNTER.with(|context| {
            let counter = context.0.get();
            assert!(!counter.is_null(), "node deallocation outside its owner");
            // SAFETY: The active counter is live and exclusively borrowed for this operation.
            unsafe {
                counter.write(
                    counter
                        .read()
                        .checked_sub(layout.size())
                        .expect("node byte underflow"),
                );
            }
            // SAFETY: The caller supplies a live Global block and the original layout.
            unsafe {
                Global.deallocate(pointer, layout);
            }
        });
    }
}

/// An ordered object whose allocator records the actual requested bytes of its nodes.
///
/// Empty maps allocate nothing. Keys and values keep their normal ownership and serialization.
/// The counter follows the map when moved, and clones receive independent counters.
pub struct ObjectMap {
    inner: BTreeMap<KeyString, Value, NodeAllocator>,
    counter: usize,
}

/// An entry in an object map.
pub enum Entry<'a> {
    /// An existing entry.
    Occupied(OccupiedEntry<'a>),
    /// A vacant entry.
    Vacant(VacantEntry<'a>),
}

/// An existing entry, carrying its map's node counter.
pub struct OccupiedEntry<'a> {
    inner: map::OccupiedEntry<'a, KeyString, Value, NodeAllocator>,
    counter: &'a mut usize,
}
/// A vacant entry, carrying its map's node counter.
pub struct VacantEntry<'a> {
    inner: map::VacantEntry<'a, KeyString, Value, NodeAllocator>,
    counter: &'a mut usize,
}
/// An iterator over the entries of an object map.
pub type Iter<'a> = map::Iter<'a, KeyString, Value>;

impl ObjectMap {
    /// Creates an empty map without allocating.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            inner: BTreeMap::new_in(NodeAllocator),
            counter: 0,
        }
    }

    /// Returns the bytes requested for tree nodes, excluding key/value payloads.
    #[must_use]
    #[inline]
    pub const fn node_storage_bytes(&self) -> usize {
        self.counter
    }

    /// Returns the number of entries.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Returns whether the map contains no entries.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Removes all entries and releases node storage and bookkeeping.
    #[inline]
    pub fn clear(&mut self) {
        if self.counter != 0 {
            with_counter(&mut self.counter, || self.inner.clear());
        }
    }
    /// Visits entries in key order.
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        self.inner.iter()
    }
    /// Visits entries in key order, with mutable values.
    #[inline]
    pub fn iter_mut(&mut self) -> map::IterMut<'_, KeyString, Value> {
        self.inner.iter_mut()
    }
    /// Visits keys in order.
    #[inline]
    pub fn keys(&self) -> map::Keys<'_, KeyString, Value> {
        self.inner.keys()
    }
    /// Visits values in key order.
    #[inline]
    pub fn values(&self) -> map::Values<'_, KeyString, Value> {
        self.inner.values()
    }
    /// Visits mutable values in key order.
    #[inline]
    pub fn values_mut(&mut self) -> map::ValuesMut<'_, KeyString, Value> {
        self.inner.values_mut()
    }
    /// Returns a value by key.
    #[inline]
    pub fn get<Q: ?Sized + Ord>(&self, key: &Q) -> Option<&Value>
    where
        KeyString: Borrow<Q>,
    {
        self.inner.get(key)
    }
    /// Returns a mutable value by key.
    #[inline]
    pub fn get_mut<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<&mut Value>
    where
        KeyString: Borrow<Q>,
    {
        self.inner.get_mut(key)
    }
    /// Returns the stored key and its value.
    #[inline]
    pub fn get_key_value<Q: ?Sized + Ord>(&self, key: &Q) -> Option<(&KeyString, &Value)>
    where
        KeyString: Borrow<Q>,
    {
        self.inner.get_key_value(key)
    }
    /// Tests whether a key is present.
    #[inline]
    pub fn contains_key<Q: ?Sized + Ord>(&self, key: &Q) -> bool
    where
        KeyString: Borrow<Q>,
    {
        self.inner.contains_key(key)
    }
    /// Inserts or replaces a value.
    #[inline]
    pub fn insert(&mut self, key: KeyString, value: Value) -> Option<Value> {
        with_counter(&mut self.counter, || self.inner.insert(key, value))
    }
    /// Removes an entry, returning its value.
    #[inline]
    pub fn remove<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<Value>
    where
        KeyString: Borrow<Q>,
    {
        with_counter(&mut self.counter, || self.inner.remove(key))
    }
    /// Removes and returns an entry.
    #[inline]
    pub fn remove_entry<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<(KeyString, Value)>
    where
        KeyString: Borrow<Q>,
    {
        with_counter(&mut self.counter, || self.inner.remove_entry(key))
    }
    /// Returns an entry for in-place insertion or modification.
    #[inline]
    pub fn entry(&mut self, key: KeyString) -> Entry<'_> {
        match with_counter(&mut self.counter, || self.inner.entry(key)) {
            map::Entry::Occupied(inner) => Entry::Occupied(OccupiedEntry {
                inner,
                counter: &mut self.counter,
            }),
            map::Entry::Vacant(inner) => Entry::Vacant(VacantEntry {
                inner,
                counter: &mut self.counter,
            }),
        }
    }
    /// Keeps entries selected by the predicate.
    #[inline]
    pub fn retain(&mut self, f: impl FnMut(&KeyString, &mut Value) -> bool) {
        with_counter(&mut self.counter, || self.inner.retain(f));
    }
    /// Returns the first entry without removing it.
    #[must_use]
    #[inline]
    pub fn first_key_value(&self) -> Option<(&KeyString, &Value)> {
        self.inner.first_key_value()
    }
    /// Returns the last entry without removing it.
    #[must_use]
    #[inline]
    pub fn last_key_value(&self) -> Option<(&KeyString, &Value)> {
        self.inner.last_key_value()
    }
    /// Removes the first entry.
    #[inline]
    pub fn pop_first(&mut self) -> Option<(KeyString, Value)> {
        with_counter(&mut self.counter, || self.inner.pop_first())
    }
    /// Removes the last entry.
    #[inline]
    pub fn pop_last(&mut self) -> Option<(KeyString, Value)> {
        with_counter(&mut self.counter, || self.inner.pop_last())
    }
    /// Iterates over a range of keys.
    #[inline]
    pub fn range<Q: ?Sized + Ord, R: RangeBounds<Q>>(
        &self,
        range: R,
    ) -> map::Range<'_, KeyString, Value>
    where
        KeyString: Borrow<Q>,
    {
        self.inner.range(range)
    }
    /// Iterates over a range of keys with mutable values.
    #[inline]
    pub fn range_mut<Q: ?Sized + Ord, R: RangeBounds<Q>>(
        &mut self,
        range: R,
    ) -> map::RangeMut<'_, KeyString, Value>
    where
        KeyString: Borrow<Q>,
    {
        self.inner.range_mut(range)
    }

    /// Moves all entries from another map, replacing duplicate keys.
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if self.is_empty() {
            std::mem::swap(self, other);
        } else if !other.is_empty() {
            self.extend(std::mem::take(other));
        }
    }

    /// Splits the map at a key, retaining separate allocation ownership in each result.
    #[must_use]
    #[inline]
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Self
    where
        KeyString: Borrow<Q>,
    {
        let mut result = Self::new();
        while self
            .last_key_value()
            .is_some_and(|(stored, _)| <KeyString as Borrow<Q>>::borrow(stored) >= key)
        {
            if let Some((key, value)) = self.pop_last() {
                result.insert(key, value);
            }
        }
        result
    }

    /// Consumes the map and yields keys in order.
    #[must_use]
    #[inline]
    pub fn into_keys(self) -> impl DoubleEndedIterator<Item = KeyString> + ExactSizeIterator {
        self.into_iter().map(|(key, _)| key)
    }
    /// Consumes the map and yields values in key order.
    #[must_use]
    #[inline]
    pub fn into_values(self) -> impl DoubleEndedIterator<Item = Value> + ExactSizeIterator {
        self.into_iter().map(|(_, value)| value)
    }
}

impl Default for ObjectMap {
    fn default() -> Self {
        Self::new()
    }
}
impl Clone for ObjectMap {
    #[inline]
    fn clone(&self) -> Self {
        let mut counter = 0;
        let inner = with_counter(&mut counter, || self.inner.clone());
        Self { inner, counter }
    }
}
impl fmt::Debug for ObjectMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}
impl PartialEq for ObjectMap {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl PartialEq<StdMap<KeyString, Value>> for ObjectMap {
    fn eq(&self, other: &StdMap<KeyString, Value>) -> bool {
        self.iter().eq(other.iter())
    }
}
impl PartialEq<ObjectMap> for StdMap<KeyString, Value> {
    fn eq(&self, other: &ObjectMap) -> bool {
        other == self
    }
}
impl Eq for ObjectMap {}
impl PartialOrd for ObjectMap {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}
impl Hash for ObjectMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}
impl<Q: ?Sized + Ord> Index<&Q> for ObjectMap
where
    KeyString: Borrow<Q>,
{
    type Output = Value;
    fn index(&self, key: &Q) -> &Value {
        &self.inner[key]
    }
}
impl Extend<(KeyString, Value)> for ObjectMap {
    fn extend<T: IntoIterator<Item = (KeyString, Value)>>(&mut self, values: T) {
        with_counter(&mut self.counter, || self.inner.extend(values));
    }
}
impl FromIterator<(KeyString, Value)> for ObjectMap {
    #[inline]
    fn from_iter<T: IntoIterator<Item = (KeyString, Value)>>(values: T) -> Self {
        let values = values.into_iter();
        let mut counter = 0;
        let inner = with_counter(&mut counter, || {
            // Bulk collection allocates a temporary Vec to sort its input.
            // Tiny objects do not need that allocation; keep bulk building for larger ones.
            if values.size_hint().1.is_some_and(|len| len <= 2) {
                let mut inner = BTreeMap::new_in(NodeAllocator);
                inner.extend(values);
                inner
            } else {
                values.collect()
            }
        });
        Self { inner, counter }
    }
}
impl<const N: usize> From<[(KeyString, Value); N]> for ObjectMap {
    fn from(values: [(KeyString, Value); N]) -> Self {
        values.into_iter().collect()
    }
}
impl From<StdMap<KeyString, Value>> for ObjectMap {
    fn from(values: StdMap<KeyString, Value>) -> Self {
        values.into_iter().collect()
    }
}
impl Serialize for ObjectMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}
impl<'de> Deserialize<'de> for ObjectMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = ObjectMap;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut access: M,
            ) -> Result<ObjectMap, M::Error> {
                let mut result = ObjectMap::new();
                while let Some((key, value)) = access.next_entry()? {
                    result.insert(key, value);
                }
                Ok(result)
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

/// An owning iterator which retains allocation bookkeeping until its last node is dropped.
pub struct IntoIter {
    inner: map::IntoIter<KeyString, Value, NodeAllocator>,
    counter: usize,
}
impl Iterator for IntoIter {
    type Item = (KeyString, Value);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        with_counter(&mut self.counter, || self.inner.next())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl DoubleEndedIterator for IntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        with_counter(&mut self.counter, || self.inner.next_back())
    }
}
impl ExactSizeIterator for IntoIter {
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl FusedIterator for IntoIter {}
impl IntoIterator for ObjectMap {
    type Item = (KeyString, Value);
    type IntoIter = IntoIter;
    fn into_iter(mut self) -> IntoIter {
        let inner = std::mem::replace(&mut self.inner, BTreeMap::new_in(NodeAllocator));
        let iter = with_counter(&mut self.counter, || inner.into_iter());
        IntoIter {
            inner: iter,
            counter: std::mem::take(&mut self.counter),
        }
    }
}
impl<'a> IntoIterator for &'a ObjectMap {
    type Item = (&'a KeyString, &'a Value);
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a mut ObjectMap {
    type Item = (&'a KeyString, &'a mut Value);
    type IntoIter = map::IterMut<'a, KeyString, Value>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(any(test, feature = "lua"))]
impl mlua::FromLua for ObjectMap {
    fn from_lua(value: mlua::Value, lua: &mlua::Lua) -> mlua::Result<Self> {
        <mlua::Table as mlua::FromLua>::from_lua(value, lua)?
            .pairs::<KeyString, Value>()
            .collect()
    }
}

impl Drop for ObjectMap {
    #[inline]
    fn drop(&mut self) {
        self.clear();
    }
}
impl Drop for IntoIter {
    #[inline]
    fn drop(&mut self) {
        with_counter(&mut self.counter, || {
            drop(std::mem::replace(
                &mut self.inner,
                BTreeMap::new_in(NodeAllocator).into_iter(),
            ));
        });
    }
}
impl fmt::Debug for IntoIter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl<'a> Entry<'a> {
    /// Returns the entry key.
    #[must_use]
    #[inline]
    pub fn key(&self) -> &KeyString {
        match self {
            Self::Occupied(e) => e.key(),
            Self::Vacant(e) => e.key(),
        }
    }
    /// Inserts a default value when vacant.
    #[inline]
    pub fn or_insert(self, value: Value) -> &'a mut Value {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(value),
        }
    }
    /// Computes a value only when vacant.
    #[inline]
    pub fn or_insert_with(self, f: impl FnOnce() -> Value) -> &'a mut Value {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(f()),
        }
    }
    /// Computes a value from its key only when vacant.
    #[inline]
    pub fn or_insert_with_key(self, f: impl FnOnce(&KeyString) -> Value) -> &'a mut Value {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => {
                let value = f(e.key());
                e.insert(value)
            }
        }
    }
    /// Modifies an existing value.
    #[must_use]
    #[inline]
    pub fn and_modify(mut self, f: impl FnOnce(&mut Value)) -> Self {
        if let Self::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }
}
impl<'a> OccupiedEntry<'a> {
    /// Returns the entry key.
    #[must_use]
    #[inline]
    pub fn key(&self) -> &KeyString {
        self.inner.key()
    }
    /// Returns the entry value.
    #[must_use]
    #[inline]
    pub fn get(&self) -> &Value {
        self.inner.get()
    }
    /// Returns a mutable entry value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut Value {
        self.inner.get_mut()
    }
    /// Returns the mutable value with the map's borrow lifetime.
    #[must_use]
    #[inline]
    pub fn into_mut(self) -> &'a mut Value {
        self.inner.into_mut()
    }
    /// Replaces the value.
    #[inline]
    pub fn insert(&mut self, value: Value) -> Value {
        with_counter(self.counter, || self.inner.insert(value))
    }
    /// Removes the value, updating the owning map's node counter.
    #[must_use]
    #[inline]
    pub fn remove(self) -> Value {
        with_counter(self.counter, || self.inner.remove())
    }
    /// Removes the key and value.
    #[must_use]
    #[inline]
    pub fn remove_entry(self) -> (KeyString, Value) {
        with_counter(self.counter, || self.inner.remove_entry())
    }
}
impl<'a> VacantEntry<'a> {
    /// Returns the entry key.
    #[must_use]
    #[inline]
    pub fn key(&self) -> &KeyString {
        self.inner.key()
    }
    /// Returns the owned key.
    #[must_use]
    #[inline]
    pub fn into_key(self) -> KeyString {
        with_counter(self.counter, || self.inner.into_key())
    }
    /// Inserts the value, updating the owning map's node counter.
    #[inline]
    pub fn insert(self, value: Value) -> &'a mut Value {
        with_counter(self.counter, || self.inner.insert(value))
    }
}

#[cfg(any(test, feature = "arbitrary"))]
impl quickcheck::Arbitrary for ObjectMap {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        StdMap::<KeyString, Value>::arbitrary(g).into()
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let entries: StdMap<_, _> = self.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        Box::new(entries.shrink().map(Self::from))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated(n: usize) -> ObjectMap {
        (0..n)
            .map(|i| (format!("field-{i:04}").into(), Value::Integer(i as i64)))
            .collect()
    }

    #[test]
    fn removal_retains_and_clear_releases_the_empty_root() {
        let mut map = ObjectMap::new();
        assert_eq!(map.node_storage_bytes(), 0);
        map.insert("key".into(), Value::Null);
        let storage = map.node_storage_bytes();
        map.remove("key");
        assert!(map.is_empty());
        assert_eq!(map.node_storage_bytes(), storage);
        map.clear();
        assert_eq!(map.node_storage_bytes(), 0);
    }

    #[test]
    fn clones_and_splits_have_independent_node_owners() {
        let mut original = populated(1000);
        let before = original.node_storage_bytes();
        let clone = original.clone();
        assert_eq!(original.node_storage_bytes(), before);
        assert_eq!(original, clone);
        drop(clone);
        assert_eq!(original.node_storage_bytes(), before);
        let mut split = original.split_off("field-0500");
        assert_eq!(original.len(), 500);
        assert_eq!(split.len(), 500);
        assert!(original.node_storage_bytes() < before);
        let retained = original.node_storage_bytes();
        split.clear();
        assert_eq!(original.node_storage_bytes(), retained);
    }

    #[test]
    fn append_moves_or_rebuilds_with_the_correct_owner() {
        let mut a = ObjectMap::new();
        let mut b = populated(100);
        let storage = b.node_storage_bytes();
        a.append(&mut b);
        assert_eq!(a.node_storage_bytes(), storage);
        assert_eq!(b.node_storage_bytes(), 0);
        b.insert("unique".into(), Value::Null);
        a.append(&mut b);
        assert_eq!(a.len(), 101);
        assert!(b.is_empty());
        let retained = a.node_storage_bytes();
        b.clear();
        assert_eq!(a.node_storage_bytes(), retained);
    }

    #[test]
    fn owned_iterator_retains_its_counter_across_threads() {
        let mut iter = populated(1000).into_iter();
        assert!(iter.next().is_some());
        assert!(iter.next_back().is_some());
        std::thread::spawn(move || drop(iter)).join().unwrap();
    }

    #[test]
    fn nested_mutation_and_entry_follow_their_own_counters() {
        let mut root = ObjectMap::from([("child".into(), Value::Object(populated(10)))]);
        let storage = root.node_storage_bytes();
        root.retain(|_, value| {
            *value = Value::Object(populated(100));
            true
        });
        assert_eq!(root.node_storage_bytes(), storage);
        let mut fresh = ObjectMap::new();
        let entry = fresh.entry("on-worker".into());
        std::thread::scope(|scope| {
            scope
                .spawn(move || {
                    entry.or_insert(Value::Object(populated(100)));
                })
                .join()
                .unwrap();
        });
        assert!(fresh.node_storage_bytes() > 0);
        let storage = fresh.node_storage_bytes();
        let value = match fresh.entry("on-worker".into()) {
            Entry::Occupied(entry) => entry.remove(),
            Entry::Vacant(_) => panic!("entry must exist"),
        };
        drop(value);
        assert_eq!(fresh.node_storage_bytes(), storage);
        fresh.clear();
        assert_eq!(fresh.node_storage_bytes(), 0);
    }

    #[test]
    fn panic_restores_the_previous_accounting_context() {
        let mut map = populated(10);
        let storage = map.node_storage_bytes();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map.retain(|_, _| panic!("predicate failed"));
        }));
        assert!(result.is_err());
        assert_eq!(map.node_storage_bytes(), storage);
        let other = populated(10);
        assert!(other.node_storage_bytes() > 0);
        assert_eq!(map.node_storage_bytes(), storage);
    }

    #[test]
    fn serde_preserves_order_values_and_duplicate_key_semantics() {
        let map: ObjectMap = serde_json::from_str(r#"{"z":1,"a":2,"z":3}"#).unwrap();
        assert_eq!(serde_json::to_string(&map).unwrap(), r#"{"a":2,"z":3}"#);
        assert!(map.node_storage_bytes() > 0);
    }
}
