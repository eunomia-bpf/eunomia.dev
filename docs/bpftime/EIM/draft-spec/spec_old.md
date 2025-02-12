# Extension Model Specification V1

## Overview

The Extension Permission and Control Model (EIM) provides a structured, formally verifiable framework to control the behavior of dynamically loaded extensions within a host environment. EIM uses *roles*, *capabilities*, *typed interfaces*, and *attributes* to define what operations an extension may perform, how it may perform them, and under what resource constraints.

Key highlights:

1. **Roles**: Assign permission bundles to extensions.
2. **Capabilities**: Define sets of allowed host APIs with well-defined type signatures and constraints.
3. **Attributes**: Impose quantitative and qualitative resource and operational limits.
4. **Typed Interfaces**: Ensure that every host and extension function adheres to specified type constraints, enabling static analysis and formal verification.
5. **Inheritance and Composition**: Support scalable, modular configuration of permissions and attributes.
6. **Formal Verification**: The model and its configurations are designed for formal verification by a verifier that checks correctness, consistency, and adherence to constraints at load time or during a pre-deployment analysis phase.

---

## Formal Language and Definitions

### Basic Sets and Domains

- Let `H` be a finite set of **Host APIs** (host-provided functions).
- Let `E` be a finite set of **Extension Exported Functions** (extension-provided entry points).
- Let `T` be a set of **Types**, each with associated constraints and properties.
- Let `C` be a set of **Capabilities**.
- Let `R` be a set of **Roles**.
- Let `A` be a set of **Attributes** applicable to roles, capabilities, and extensions. Can be used to check the compatibility or applicability between Capabilities and Roles.

We introduce a specification language that can be represented in YAML/JSON or a domain-specific language (DSL). The semantics remain the same.

### Types and Constraints

**Definition (Type):**  
A type `τ ∈ T` is defined as `(BaseType, Constraints)`, where:
- `BaseType` is a primitive (e.g., `int`, `size_t`, `char*`) or composite (e.g., struct) type.
- `Constraints` is a finite set of logical predicates over values of that type. Examples:
  - Non-null pointer: `value ≠ NULL`
  - Bounded integer: `0 ≤ value ≤ 100`
  - Null-terminated string: `∀ i, value[i] ≠ '\0'` until a terminating null is found

**Type System Requirements:**
- For each `τ ∈ T`, `Constraints(τ)` must be decidable predicates, allowing static or symbolic verification.
- The verifier can check, for every function call, that arguments and return values respect these constraints.

**Example Type Definitions:**
```yaml
types:
  - name: "const_char_ptr"
    base: "char*"
    constraints: ["non_null", "null_terminated"]

  - name: "int_positive"
    base: "int"
    constraints: ["value ≥ 0"]
```

### Host APIs

**Definition (Host Function):**  
Each host function `h ∈ H` is defined as:
```
h = (Name_h, (τ1, τ2, ..., τn) -> τ_out, F_Constraints)
```
- `Name_h`: A unique identifier (string).
- `(τ1, τ2, ..., τn)`: Parameter types.
- `τ_out`: Return type.
- `F_Constraints`: A set of logical conditions specifying preconditions, postconditions, and invariants. For example, if `τ_out` is `int` representing a file descriptor, `F_Constraints` might state: "If return < 0, error occurred; else resource is valid." Or, the constraints can define the relationship between the arguments, such as the first argument is a non-null pointer to a null-terminated string, the second argument is the size of the buffer.
- `Attributes`: Some host functions may have attributes if they have side effects. For example, create a file will change the disk, or make a network call will change the network state. They have attributes like `may_have_side_effect = true`.

**The verifier** checks:
- That each call made by the extension to `h` satisfies `F_Constraints`.
- Argument and return types match the signature and type constraints.
- For some host functions, the verifier also check and record the resource allocation and deallocation. So we can identify the leak of some resources.

Example in C:
```c
int host_open_file(const char *filename, const char *mode);
void* host_malloc(size_t size);
void host_free(void* ptr);
```

### Extension Exported Functions

Similarly, extension functions `e ∈ E` are defined:
```
e = (Name_e, (τ1, τ2, ..., τm) -> τ_out, E_Constraints)
```
- The host may call these functions.
- The verifier ensures these also adhere to their constraints.

example in C:
```c
UPROBE_ENTRY(extension_entry, (int, int) -> void)
{
    ...
}
```

### Capabilities

**Definition (Capability):**  
A capability `c ∈ C` is defined as:
```
c = (Name_c, H_c, Cap_Constraints)
```
- `Name_c`: Unique capability name.
- `H_c ⊆ H`: A subset of host APIs exposed to the extension.
- `Cap_Constraints`: Additional constraints or policies applying to these APIs as a group. For example, `Cap_Constraints` might restrict usage frequency or data size.

**Capability Composition:**
Given `c1 = (Name_c1, H_c1, CapCon1)` and `c2 = (Name_c2, H_c2, CapCon2)`, a composed capability `c_comp` is:
```
c_comp = (Name_comp, H_c1 ∪ H_c2, CapCon_comp)
```
`CapCon_comp` must be derived so that all constraints remain satisfiable. If any constraints conflict (e.g., one requires read-only and the other requires write access), composition fails unless explicitly resolved by a policy. The verifier checks logical consistency of combined constraints.

### Attributes

Attributes `A` are key-value pairs that impose resource and operational constraints:
- `A.max_memory`: Maximum memory (in bytes) the extension can use.
- `A.max_instructions`: Maximum instruction count per invocation.
- `A.allow_memory_write`: Boolean indicating if extension can write to host memory (if allowed by host APIs).
- `A.network_access`: Boolean controlling network operations.
- Other domain-specific attributes as required.

The verifier ensures that attribute constraints are not violated by the extension’s code or by the chosen capabilities. For instance, if `allow_memory_write = false` but a chosen capability grants `host_write_mem`, verification fails or the extension is denied loading.

### Roles

**Definition (Role):**
```
R = (Name_R, Capabilities_R, Attributes_R, Parents_R)
```
Where:
- `Name_R`: Unique role name.
- `Capabilities_R ⊆ C`: The set of capabilities granted by this role.
- `Attributes_R`: A mapping from attribute keys to values.
- `Parents_R ⊆ R`: Zero or more parent roles from which `R` inherits.

**Inheritance Rules:**
- Capabilities: Union of capabilities from parents and this role. The verifier checks for conflicts in composed capabilities.
- Attributes: If parents define the same attribute differently, the child must explicitly resolve the conflict or the verifier rejects the role. Attributes are merged following a deterministic policy (e.g., the child’s explicit setting overrides parent settings).
- Types and constraints from parents must not be weakened. Contradictions in type constraints result in a verification error.

The final effective permissions of a role `R_eff` are:
```
R_eff = (Name_R, Union of all parent capabilities + self, Resolved_Attributes, ...)
```
The verifier processes the inheritance graph to produce a conflict-free, fully expanded set of capabilities and attributes.

### Assigning Roles to Extensions

When loading an extension `X`, the host assigns one or more roles `R_assigned`. The effective permissions of `X` are the intersection of constraints imposed by all these roles combined. If any conflict arises, verification fails and the host rejects the extension.

---

## Formal Verification Approach

A **verifier** tool is run at deployment or load time. It checks:

1. **Type Consistency**:  
   For each host API in each capability, and each extension-exported function, the verifier confirms that all arguments and return values satisfy the declared type constraints.  
   If `int_positive` is required and the extension code can produce negative integers (detected via static analysis or symbolic execution), verification fails.

2. **Attribute Compliance**:  
   The verifier analyzes instructions, memory usage patterns (statically approximated), and API calls to ensure they do not exceed the specified attributes.  
   For example, if `max_memory = 1048576`, the verifier ensures that no sequence of valid API calls can cause allocation beyond 1MB of memory. This may rely on conservative static analysis and symbolic constraints.

3. **Capability and Role Composition**:
   The verifier checks that all composed capabilities and inherited roles produce a logically consistent set of constraints.  
   - If two parent roles define contradictory constraints for the same API, verification fails.
   - If a composed capability is formed from `FileRead` and `FileWrite`(They have attributes like `read_only = true` and `read_only = false` respectively) but has contradictory `read_only` constraints, verification fails or requires explicit resolution.

4. **Non-functional Properties** (optional):  
   The verifier can also check additional invariants like no possible null dereferences if `non_null` is specified, or no buffer overflows if length constraints are declared.

**Verification Algorithmic Aspects:**
- The verifier may use a combination of:
  - **Type checking**: Ensuring types match declared signatures.
  - **Symbolic execution**: Exploring code paths in the extension to ensure no violation of constraints.
  - **Policy-driven resolution**: If multiple parents define the same attribute, the verifier enforces a predefined priority.

If verification succeeds, the host loads the extension with full confidence that the declared roles and capabilities are safe and consistent. If verification fails, the loading is aborted.

---

## Examples

### Example 1: Simple Roles and Capabilities

```yaml
types:
  - name: "const_char_ptr"
    base: "char*"
    constraints: ["non_null", "null_terminated"]
    
  - name: "int_positive"
    base: "int"
    constraints: ["value ≥ 0"]

host_functions:
  - name: "host_open_file"
    signature:
      params:
        - {type: "const_char_ptr", name: "filename"}
        - {type: "const_char_ptr", name: "mode"}
      return_type: "int"
    function_constraints:
      - "filename.non_null"
      - "mode.non_null"
      - "return ≥ -1"
      - "If return < 0, indicates an error."

capabilities:
  - name: "FileRead"
    apis: ["host_open_file", "host_read_file"]
    constraints: ["All returned file descriptors must be checked before use"]
  
  - name: "FileWrite"
    apis: ["host_write_file"]

roles:
  - name: "BasicFileRole"
    capabilities: ["FileRead"]
    attributes:
      max_memory: 1048576
      allow_memory_write: false
      max_instructions: 1000000
    inherits: []

  - name: "ExtendedFileRole"
    capabilities: ["FileWrite"]
    # Inherits FileRead from BasicFileRole
    inherits: ["BasicFileRole"]

# The verifier checks:
# - No conflicts in capabilities.
# - Attributes inherited properly.
# If all checks pass, ExtendedFileRole effectively grants host_open_file, host_read_file, host_write_file, with specified attributes.
```

### Example 2: Verifying Type Constraints

Suppose the extension exports:
```yaml
extension_exports:
  - name: "process_data"
    signature:
      params:
        - {type: "int_positive", name: "count"}
      return_type: "const_char_ptr"
    constraints:
      - "If return ≠ NULL, must be a null_terminated string"
```
The verifier ensures:
- The extension’s code never passes a negative integer to `count`.
- The returned pointer is always valid and null-terminated if not NULL.

If verification finds a path where `count` could be negative, loading is rejected.

---

## Conclusion

This specification describes a comprehensive and formally verifiable permission and control model for extensions. By integrating typed interfaces, capabilities, attributes, roles, and a formal verification process, it ensures a robust, secure, and analyzable framework suitable for production systems and academic research.

The model’s modular design, combinable capabilities, and inheritable roles provide flexibility. The rigorous typing and verification approach ensures that any permitted operation is safe, well-defined, and consistent with stated constraints, reducing runtime errors and security vulnerabilities.
