`ObjectMap` now records its tree-node allocation size and exposes `node_storage_bytes()`. Its Rust type is an allocator-aware map instead of `std::collections::BTreeMap`; VRL object ordering and serialization are unchanged.

authors: sakateka
