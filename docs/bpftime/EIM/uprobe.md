# UProbe Role Definition README


This example assumes:

- We have a set of host APIs (BPF helpers) related to uprobe usage available.
- We define capabilities for these APIs.
- We assign attributes that control resource and operational constraints.
- We treat the uprobe in userspace eBPF as a specific Role that grants access to the relevant host APIs.

## Introduction

In the model, a **Role** defines which capabilities (and thus host APIs) and attributes an extension can access. If we consider **uProbes** in eBPF as one such role, we need to define:

1. **Host APIs (BPF Helpers) for uProbes**: For example:
   - Map helpers: `bpf_map_lookup_elem`, `bpf_map_update_elem`, `bpf_map_delete_elem`
   - String helpers: `bpf_strncpy`, `bpf_strnlen`, `bpf_strcmp`
   - Memory and probe helpers: `bpf_probe_read`, `bpf_probe_read_str`
   - Info helpers: `bpf_get_current_pid_tgid` (or hypothetical helpers that retrieve process/thread info)
   
2. **Capabilities**: Group these helpers into logically coherent sets that reflect common usage patterns. For instance:
   - `MapHelpersCap` for map operations
   - `StringHelpersCap` for string operations
   - `ProbeHelpersCap` for reading memory and retrieving info

3. **Attributes**: Define resource and behavioral constraints for running uProbes, such as:
   - `max_memory` (limit memory usage)
   - `max_instructions` (limit execution time per event)
   - `allow_memory_write` (does this role allow modifications to host memory? Usually not)
   - `always_const`, `may_have_side_effect`, `skip_verification` (behavioral attributes depending on trust level)

By composing these capabilities and setting attributes, we create a `UprobeRole` that extensions can assume to safely and predictably perform uprobe logic.

---

## Example Role Definition (Conceptual)

```yaml
types:
  - name: "const_char_ptr"
    base: "char*"
    constraints: ["non_null", "null_terminated"]

  - name: "int_positive"
    base: "int"
    constraints: ["value ≥ 0"]

host_functions:
  # Map Helpers
  - name: "bpf_map_lookup_elem"
    signature:
      params:
        - {type: "void_ptr", name: "map"}
        - {type: "const_void_ptr", name: "key"}
      return_type: "void_ptr"
    function_constraints:
      - "map.non_null"
      - "key.non_null"

  - name: "bpf_map_update_elem"
    signature:
      params:
        - {type: "void_ptr", name: "map"}
        - {type: "const_void_ptr", name: "key"}
        - {type: "const_void_ptr", name: "value"}
        - {type: "int", name: "flags"}
      return_type: "int"
    function_constraints:
      - "map.non_null"
      - "key.non_null"
      - "value.non_null"

  - name: "bpf_map_delete_elem"
    signature:
      params:
        - {type: "void_ptr", name: "map"}
        - {type: "const_void_ptr", name: "key"}
      return_type: "int"

  # String Helpers
  - name: "bpf_strncpy"
    signature:
      params:
        - {type: "char_ptr", name: "dest"}
        - {type: "const_char_ptr", name: "src"}
        - {type: "size_t", name: "len"}
      return_type: "char_ptr"
    function_constraints:
      - "dest.non_null"
      - "src.non_null"

  - name: "bpf_strnlen"
    signature:
      params:
        - {type: "const_char_ptr", name: "str"}
        - {type: "size_t", name: "maxlen"}
      return_type: "size_t"
    function_constraints:
      - "str.non_null"

  # Probe Helpers
  - name: "bpf_probe_read"
    signature:
      params:
        - {type: "void_ptr", name: "dst"}
        - {type: "size_t", name: "len"}
        - {type: "const_void_ptr", name: "unsafe_ptr"}
      return_type: "int"
    function_constraints:
      - "dst.non_null"
      - "len > 0"

  # Info Helpers
  - name: "bpf_get_current_pid_tgid"
    signature:
      params: []
      return_type: "int"
    function_constraints:
      - "return ≥ 0"

  # Uprobe event handling
  - name: "handle_uprobe_event"
    signature:
      params:
        - {type: "const_void_ptr", name: "ctx"}
      return_type: "int"
    function_constraints:
      - "ctx.non_null"

capabilities:
  - name: "MapHelpersCap"
    apis: ["bpf_map_lookup_elem", "bpf_map_update_elem", "bpf_map_delete_elem"]
    constraints: ["All map operations require non_null map and key"]

  - name: "StringHelpersCap"
    apis: ["bpf_strncpy", "bpf_strnlen"]
    constraints: ["String pointers must be non_null"]

  - name: "ProbeHelpersCap"
    apis: ["bpf_probe_read", "bpf_get_info"]
    constraints: ["Memory read must be from safe address", "Info retrieval must return non-negative"]

  - name: "UprobeEventCap"
    apis: ["handle_uprobe_event"]
    constraints: ["ctx must be valid and non_null"]

roles:
  - name: "UprobeRole"
    capabilities: ["MapHelpersCap", "StringHelpersCap", "ProbeHelpersCap", "UprobeEventCap"]
    attributes:
      max_memory: 512                # 512 bytes stack limit
      max_instructions: 100000       # max instructions per event
      allow_memory_write: false      # not writing host memory
      always_const: true            # All variables from extension arguments are treated as const even if not declared as const
      may_have_side_effect: false     # All resources allocated by the extension are freed by the extension
      skip_verification: false       # must be verified

extension_exports:
  # Extensions assigned to UprobeRole can define their own entry points that
  # the host calls when the uprobe fires.
  # For all global functions, the extension can treat them as its own entry points.
```

---

## Notes

- **Host APIs (uFunc/helpers)**: We define each BPF helper as a host function with type constraints. The `map helpers`, `bpf_string` helpers, `bpf_probe_read`, and `bpf_get_info` are part of the defined host APIs. `handle_uprobe_event` is also defined, representing a host function that can be called by the extension when a uprobe event occurs.
  
- **Capabilities**: We group related APIs into capabilities. For instance, `MapHelpersCap` contains all the map-related functions. This modular approach allows reusing capabilities in other roles if needed.

- **Attributes**: We set `max_memory`, `max_instructions`, and other attributes to suit a typical uprobe scenario. `may_have_side_effect = true` acknowledges that reading memory or updating maps can influence system state. `skip_verification = false` means the verifier must run to ensure safety.

- **UprobeRole**: The `UprobeRole` aggregates these capabilities and attributes. An extension assigned this role can use the listed APIs and must respect the defined constraints.

- **Extension Exported Function**: The extension can define its own entry points that the host calls when the uprobe fires. For all global functions, the extension can treat them as its own entry points. The global functions types will be resolved from BTF.

