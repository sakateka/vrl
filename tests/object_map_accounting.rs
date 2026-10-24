#![cfg(feature = "value")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicIsize, Ordering::Relaxed},
};

use vrl::value::{KeyString, ObjectMap, Value};

struct Oracle;
static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: Every allocation is delegated to System with its original layout.
unsafe impl GlobalAlloc for Oracle {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid nonzero layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size().cast_signed(), Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller supplies a live System block and its original layout.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size().cast_signed(), Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: Oracle = Oracle;

fn map_bytes(map: &ObjectMap) -> usize {
    map.node_storage_bytes()
        + map
            .iter()
            .map(|(key, value)| key.capacity() + heap_bytes(value))
            .sum::<usize>()
}

// The fixture uses only owned trees, integer values and arrays. Shared byte
// backing storage and allocator rounding are outside this measurement.
fn heap_bytes(value: &Value) -> usize {
    match value {
        Value::Object(map) => map_bytes(map),
        Value::Array(values) => {
            values.capacity() * size_of::<Value>() + values.iter().map(heap_bytes).sum::<usize>()
        }
        Value::Integer(_) | Value::Null => 0,
        _ => unreachable!("unexpected fixture value"),
    }
}

fn populated(count: usize) -> ObjectMap {
    (0..count)
        .map(|i| {
            let mut values = Vec::with_capacity(17);
            values.push(Value::Integer(i64::try_from(i).unwrap()));
            (format!("field-{i:04}").into(), Value::Array(values))
        })
        .collect()
}

#[test]
fn accounting_matches_live_allocator_requests_after_ownership_changes() {
    // Keep this oracle in a separate integration-test binary so other tests
    // cannot allocate concurrently with the measured ownership transitions.
    let baseline = LIVE.load(Relaxed);
    let mut first = populated(1000);
    assert_eq!(
        map_bytes(&first),
        (LIVE.load(Relaxed) - baseline).cast_unsigned()
    );

    let clone_baseline = LIVE.load(Relaxed);
    let clone = first.clone();
    assert_eq!(
        map_bytes(&clone),
        (LIVE.load(Relaxed) - clone_baseline).cast_unsigned()
    );
    drop(clone);

    first.retain(|key, _| key.as_str() >= "field-0500");
    assert_eq!(
        map_bytes(&first),
        (LIVE.load(Relaxed) - baseline).cast_unsigned()
    );
    let mut second = first.split_off("field-0750");
    assert_eq!(
        map_bytes(&first) + map_bytes(&second),
        (LIVE.load(Relaxed) - baseline).cast_unsigned()
    );
    first.append(&mut second);
    assert_eq!(
        map_bytes(&first) + map_bytes(&second),
        (LIVE.load(Relaxed) - baseline).cast_unsigned()
    );
    std::mem::swap(&mut first, &mut second);
    first.clear();
    second.retain(|_, _| false);
    assert!(second.is_empty());
    assert_eq!(
        map_bytes(&second),
        (LIVE.load(Relaxed) - baseline).cast_unsigned()
    );
    drop((first, second));
    assert_eq!(LIVE.load(Relaxed), baseline);

    let mut map = ObjectMap::new();
    map.insert("nested".into(), Value::Object(populated(100)));
    assert_eq!(
        map_bytes(&map),
        (LIVE.load(Relaxed) - baseline).cast_unsigned()
    );
    let mut iter = map.into_iter();
    drop(iter.next());
    drop(iter);
    assert_eq!(LIVE.load(Relaxed), baseline);

    // Check both sides of inline storage and adopted over-capacity Strings.
    for text in [
        "",
        "short",
        "123456789012345678901234",
        "1234567890123456789012345",
        "ключ-ключ-ключ-ключ-ключ",
    ] {
        let before = LIVE.load(Relaxed);
        let key = KeyString::from(text);
        assert_eq!(
            key.capacity(),
            (LIVE.load(Relaxed) - before).cast_unsigned()
        );
        let cloned = key.clone();
        assert_eq!(key.as_str(), cloned.as_str());
        assert_eq!(
            key.capacity() + cloned.capacity(),
            (LIVE.load(Relaxed) - before).cast_unsigned()
        );
        drop((key, cloned));
        assert_eq!(LIVE.load(Relaxed), before);

        let mut string = String::with_capacity(4096);
        string.push_str(text);
        let key = KeyString::from(string);
        assert_eq!(
            key.capacity(),
            (LIVE.load(Relaxed) - before).cast_unsigned()
        );
        assert_eq!(key.as_str(), text);
        drop(key);
        assert_eq!(LIVE.load(Relaxed), before);
    }
}
