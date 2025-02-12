# Extension Inferface Model Specification V3

---

## 1. Scope

This specification defines a model (EIM) for controlling the loading and execution of code extensions within a host environment. The model ensures that such extensions operate safely, within defined constraints, and only invoke allowed host-provided functions under controlled conditions.

EIM includes six key concepts:

- **Types**: Ensure data correctness and safe usage patterns.
- **Extension-Exported Entries**: Functions exposed by the extension, serving as hooks or callbacks for the host.
- **Host-Provided Functions (Helpers)**: Host functions callable by the extension, subject to verification and constraints.
- **Capabilities**: A set of host-provided functions grouped with constraints, serving as permission bundles.
- **Roles**: A set of Extension-Exported Entries plus constraints that govern their usage and indirectly control capabilities
- **Constraints**: Conditions that ensure resource limits, operational restrictions, and safe execution at every level of the model.

These concepts collectively enable a principle-of-least-privilege approach and formal verification of extension correctness and safety.

## 2. Normative References

The normative language used herein ("MUST", "MUST NOT", "SHOULD", "SHOULD NOT") is as defined in RFC 2119. Unless otherwise noted, all statements are normative requirements.

## 3. Terms and Definitions

- **Extension**: Dynamically loaded code that the host can invoke at defined hook points, analogous to an eBPF program or a plugin.
- **Host Environment**: The system or runtime that loads the extension and provides host-provided functions (helpers).

## 4. Key Concepts

### 4.1 Types

**Definition:** A **Type** associates a base data type (e.g., `int`, `char*`) with a set of constraints that values of that type must satisfy. Constraints on types (type-level constraints) can include properties such as `non_null`, `null_terminated`, or `value ≥ 0`.

**Definition (Type):**

A type `τ ∈ T` is defined as `(BaseType, Constraints(τ))`, where:

- `BaseType` is a primitive or composite data type (e.g., `int`, `size_t`, `char*`, or a struct).
- `Constraints(τ)` is a finite set of logical predicates that must hold for any value of that type (e.g., non_null, bounded integer, null_terminated).

**Type System Requirements:**

- For each `τ ∈ T`, `Constraints(τ)` must be decidable, enabling static or symbolic verification.
- The verifier checks that arguments and return values of all functions (hooks and helpers) adhere to their declared type constraints.

**Example:**

```yaml
types:
  - name: "int_positive"
    base: "int"
    constraints: ["value ≥ 0"]
  - name: "const_char_ptr"
    base: "char*"
    constraints: ["non_null", "null_terminated"]
```

### 4.2 Extension-Exported Entries

An **Extension-Exported Entry** is a function provided by the extension that the host can call. It defines an interface point (hook) at which the host invokes the extension’s logic.

**Definition**:

```makefile
e = (Name_e, (τ1, τ2, ..., τm) -> τ_out, E_Constraints)
```

- The host invokes these hooks.
- Hooks must adhere to all declared constraints (type-level and function-level).

**Requirements:**

- Each extension-exported entry MUST declare its parameter and return types, and any function-level constraints.
- If the extension-exported entry’s constraints or types are violated at runtime, verification MUST fail and the extension MUST NOT be loaded or executed.

### **Example:**

```yaml
extension_exports:
  - name: "process_data"
    signature:
      params:
        - { type: "int_positive", name: "count" }
      return_type: "int_positive"
    constraints:
      - "return ≥ 0"
```

And uprobe/uretprobe hooks

One Extension-Exported Entries(hook) means it has the same type and execution model.

### 4.3 Host-Provided Functions (Helpers)

**Host-Provided Functions (Helpers)** are functions offered by the host environment that the extension may call if allowed by the assigned role and capabilities. These enable the extension to interact with host resources (e.g., reading files, sending packets) under controlled conditions.

**Definition (Host Function):**

Each **Host-Provided** `h ∈ H` is defined as:

```makefile
h = (Name_h, (τ1, τ2, ..., τn) -> τ_out, F_Constraints)
```

- `(τ1, τ2, ..., τn)`: Parameter types subject to type constraints.
- `τ_out`: Return type subject to type constraints.
- `F_Constraints`: Additional function-level constraints (e.g., write memory, side-effect markers).

**Verifier Checks for Host APIs:**

- Ensures arguments and returns satisfy type constraints.
- Checks function-level constraints, such as side effects, resource allocation, or error signaling.

**Requirements:**

- Each helper MUST define its signature in terms of types and may have additional function-level constraints.
- If an extension attempts to call a helper with arguments violating its type-level or function-level constraints, the verifier MUST detect this and prevent loading or execution.

**Example (Host-Provided Function):**

```yaml
host_functions:
  - name: "host_read_file"
    signature:
      params:
        - { type: "int_positive", name: "fd" }
      return_type: "int"
    function_constraints:
      - "return ≥ -1"
```

**Example:**

```c
// Constraints: filename.non_null, mode.non_null, return ≥ -1,
int host_open_file(const char *filename, const char *mode);
```

If `fd` is negative, calling `host_read_file` violates `int_positive` constraints, causing verification failure.

---

### 4.4. Capabilities

**Definition:**

```makefile
Capability = (Name_c, H_c, Cap_Constraints)
```

- `Name_c`: The capability's name.
- `H_c`: A subset of Host-Provided Functions included in this capability.
- `Cap_Constraints`: Constraints applicable to these helpers as a group (e.g., `read_only = true`).

**Requirements:**

- A capability groups a set of helpers and may impose additional constraints on their usage.
- The extension may only use helpers provided by capabilities assigned to its role.
- If a capability states conditions like `read_only = true`, then adding a write  helpers to the  Capability violates the constraints, causing verification failure.

The Cap_Constraints can be from the F_Constraints, and need to verify:

1. if Capability  were missing `read_only`, adding a write helpers means add `read_only`  = false to Capability 
2. If Capability  has `read_only` == false, you can add a write helpers to it.
3. If Capability  has `read_only` == true, you cannot add a write helpers to it.

**Example:**

```yaml
capabilities:
  - name: "FileRead"
    apis: ["host_read_file"]
    constraints:
      - "read_only = true"

```

`FileRead` capability restricts the extension to read-only file operations.

---

### 4.5. Roles

A Role can be understood as a selection of extension-exported entries governed by a set of constraints.

**Definition:**

```makefile
Role = (Name_R, Exported_R, Constraints_R)
```

- `Name_R`: The role's name.
- `Exported_R ⊆ Extension-Exported Entries`: The set of extension-exported functions (hooks) that this role makes available.
- `Constraints_R`: governing not only resource limits and operational policies, but also the implicit availability or prohibition of certain capabilities. These constraints can reference capabilities by name, effectively linking which host-provided functions the extension can use through these capabilities.

A role can be viewed as primarily a set of extension-exported entries (i.e., the functions the extension provides to the host) combined with a collection of constraints that control what the extension can do. Through these constraints, a role can effectively allow or disallow the use of specific capabilities. In other words, while the role does not explicitly contain a list of capabilities, its constraints can mention capabilities by name (e.g., use_capability: FileRead) or forbid them (e.g., !NetAccess). This approach maintains a minimal definition of a role while still enabling fine-grained control over the extension’s allowed helpers and operations.

**Note on Inheritance:**

Optionally, a role may inherit from another role, merging constraints. If inheritance is used, conflicting constraints must be resolved. This feature is not essential to the core model.

**Requirements:**

- A role primarily determines which extension hooks are available and under what conditions capabilities (and thus helpers) can be used.
- The role’s constraints MUST be satisfied for the extension to run.

**Example of Role-Level Constraints Interacting with Capabilities:**

Suppose we have two capabilities:

```yaml
capabilities:
  - name: "NetAccess"
    apis: ["host_send_packet"]
    constraints:
      - "side_effects = network"

  - name: "FileRead"
    apis: ["host_read_file"]
    constraints:
      - "read_only = true"
```

A role might look like:

```yaml
roles:
  - name: "DataProcRole"
    exported: ["process_data"]
    constraints:
      - "memory_usage ≤ 65536"
      - "side_effects = false"
      - "use_capability: FileRead"
      - "!NetAccess"   # This forbids the NetAccess capability.
```

Here:

- `side_effects = false` is a constraint that forbids any helpers with `side_effects = t`rue.
- `use_capability: FileRead` states that the `FileRead` capability is allowed, granting access to `host_read_file`.
- `!NetAccess` explicitly forbids the `NetAccess` capability, so even if some global policy would have allowed it, the role disallows it.

This demonstrates how `Constraints_R` can define a relationship between capabilities and a role. By using `!Capability`, we forbid certain capabilities; by referencing the capability by name (e.g., `use_capability: FileRead`), we allow it.

---

### 4.6. Constraints

**Definition:** **Constraints** are logical conditions that MUST hold at various levels: type, function, capability, and role. They unify resource limitations, operational policies, type rules, and side-effect restrictions.

**Requirements:**

- Constraints must be decidable and enforceable by the verifier.
- Any violation of constraints (type-level, function-level, capability-level, or role-level) MUST cause verification to fail.

**Examples of Constraints:**

- Type-level: `value ≥ 0`
- Function-level (helper or extension entry): `return ≥ -1`
- Capability-level: `read_only = true`
- Role-level: `side_effect`, `memory_usage ≤ 65536`, `!NetAccess`

---

## Verification Process

The verifier checks the extension against the assigned role:

1. **Type Consistency**: Ensures all parameters and returns align with their type constraints.
2. **Constraint Compliance**: Ensures all constraints at all levels (type, function, capability, role) are satisfied.
3. **Capability and Role Alignment**: Ensures the extension only uses helpers allowed by capabilities permitted in the role. Any forbidden capabilities (e.g., `!NetAccess`) or actions (`side_effect`) result in verification failure.
4. **Error Handling**: If a helper returns a value that must be checked (e.g., a file descriptor that could be negative) and the extension fails to verify it before using it with another helper demanding a positive value, verification fails.

If verification passes, the extension loads safely. Otherwise, it is rejected.

---

## OOP Analogy (Non-Normative)

While EIM is not object-oriented, one may draw a conceptual analogy:

- **Roles**: Similar to classes, defining a blueprint of what extension entries are exposed and what capabilities and constraints apply.
- **Extensions**: Similar to objects (instances of roles). Assigning a role to an codebase is like creating an object from a class. The extension code (like object methods) to be loaded must adhere to constraints (like class invariants).
- **Types, Constraints, and Capabilities**: Analogous to type checks, resource policies, and method sets in a class.

This analogy is informal and only intended to help conceptual understanding.

---

## Conclusion

This specification provides a formal framework where **Types**, **Extension-Exported Entries**, **Host-Provided Functions (Helpers)**, **Capabilities**, **Roles**, and **Constraints** work together to ensure safe, controlled execution of dynamically loaded extensions. The model emphasizes minimal definitions, strong verification steps, and least-privilege design.
