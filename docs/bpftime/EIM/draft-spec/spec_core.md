
## Extension Interface Model (EIM) V2

### 1. Introduction

The Extension Interface Model (EIM) is a specification framework that defines how a host environment can describe, verify, and regulate **extensions**—code modules that integrate with a host application or system to provide additional functionality. Such extensions may be introduced at various stages, including compile-time, link-time, load-time, or even dynamically at runtime. EIM ensures that each extension operates within rigorously defined boundaries, adheres to declared safety constraints, and invokes only authorized host-provided interfaces.

By specifying types, capabilities, constraints, and extension interface points (entries), EIM imposes a least-privilege model that enables safe integration without sacrificing flexibility or expressiveness. The EIM is independent of any particular runtime technology or programming language: implementers can define and enforce EIM specifications using different verification mechanisms and execution environments.

### 2. Key Concepts

- **Types**: Typed parameters and return values, each constrained by conditions ensuring safe usage (e.g., non-null, resource-limited).
- **Extension-Exported Entries (Extension Functions)**: Functions exposed by the extension that the host can call into. They define explicit contracts for behavior, preconditions, and postconditions.
- **Host-Provided Functions (Helpers)**: Functions or interfaces provided by the host and selectively available to extensions. Helpers grant controlled access to system resources and must be explicitly authorized.
- **Capabilities**: Named permission bundles that group sets of allowed helpers and impose associated constraints, enabling fine-grained control over what an extension can do.
- **Constraints**: Logical conditions attached to types, capabilities, and interfaces that ensure the extension’s behavior remains safe, correct, and predictable.

The EIM specification is a framework for system designers: it is not a one-size-fits-all solution but rather a customizable blueprint that can be tuned to balance safety, performance, and expressiveness. Designers define their own types, constraints, capabilities, and extension entries according to their specific application domain and risk profile.

### 3. Normative References

The terms "MUST", "MUST NOT", "SHOULD", and "SHOULD NOT" are interpreted as defined in [RFC 2119].

Unless explicitly stated otherwise, all statements in this document are normative requirements.

### 4. Terminology

- **Extension**: A code module integrated with a host, which may be introduced at various integration points (e.g., at build-time, through static linking, or dynamically at runtime).
- **Host Environment**: The encompassing runtime or system that loads or links the extension and enforces EIM rules. The host environment provides helpers and a verification mechanism.
- **Verifier**: A component of the host environment that checks an extension’s compliance with the EIM specification before allowing execution.

### 5. Types

A **Type** `τ` is `(BaseType, Constraints(τ))`, where:

- **BaseType**: A primitive or composite data type (e.g., `int`, `size_t`, `char*`, or a specific struct).
- **Constraints(τ)**: A finite set of verifiable conditions (e.g., `value ≥ 0`, `non_null`).

**Requirements:**

- Type constraints MUST be statically or symbolically checkable to ensure safety and correctness before the extension executes.
- All parameters and return values in extension entries and helpers MUST define their type and adhere to the associated constraints.

**Example:**

```yaml
types:
  - name: "int_positive"
    base: "int"
    constraints:
      - "value ≥ 0"
  - name: "const_char_ptr"
    base: "char*"
    constraints:
      - "non_null"
      - "null_terminated"
```

### 6. Constraints

Constraints enforce safe, correct, and resource-limited behavior.

**Two Classes of Constraints:**

1. **Relational Constraints**: Express logical relationships and resource bounds (e.g., `value ≥ 0`, `memory_usage ≤ 65536`).
2. **Tag-Based Constraints**: Mark semantic properties or operational modes (e.g., `read_only = true`, `side_effects = false`).

**Requirements:**

- Constraints MUST be decidable.
- Violating any constraint at verification time MUST prevent the extension from running.
- Constraints may encode functional correctness, resource limits, security policies, or side-effect restrictions.

### 7. Host-Provided Functions (Helpers)

**Helpers** are controlled entry points to host functionality. They represent operations like reading from a file, allocating memory, or sending a network packet. Access to these functions is restricted by capabilities.

**Definition:**

```makefile
h = (Name_h, (τ1, τ2, ..., τn) -> τ_out, Helper_Constraints)
```

- `(τ1, τ2, ..., τn)`: Parameter types, each with constraints.
- `τ_out`: A return type with type-level constraints.
- `Helper_Constraints`: Additional pre/postconditions (e.g., `return ≥ -1`).

**Requirements:**

- Each helper MUST specify its signature and constraints.
- The verifier MUST reject extensions that violate these constraints.

**Example:**

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

### 8. Capabilities

**Capabilities** bundle sets of helpers and impose overarching constraints. By assigning capabilities to an extension entry, the host grants controlled access to specific functionalities.

**Definition:**

```makefile
Capability = (Name_c, H_c, Cap_Constraints)
```

- `Name_c`: The capability’s name.
- `H_c`: The set of host-provided functions included in this capability.
- `Cap_Constraints`: Constraints applying to all helpers in this capability.

**Requirements:**

- The extension can only invoke helpers from capabilities it is allowed to use.
- Conflicting constraints within a capability’s definition MUST cause verification to fail.

**Example:**

```yaml
capabilities:
  - name: "FileRead"
    apis: ["host_read_file"]
    constraints:
      - "read_only = true"
```

### 9. Extension-Exported Entries (Extension Functions)

Extension-exported entries are the callable interfaces through which the host invokes the extension’s functionality. Each entry declares its input/output types, constraints, and allowed capabilities.

**Definition:**

```makefile
e = (Name_e, (τ1, τ2, ..., τm) -> τ_out, E_Constraints, Allowed_Capabilities)
```

- `Name_e`: Name of the extension-exported entry.
- `(τ1, τ2, ..., τm) -> τ_out`: Parameter and return types, each with defined constraints.
- `E_Constraints`: Additional constraints relevant to this entry.
- `Allowed_Capabilities`: The set of capabilities (and thus helpers) this entry may use.

**Requirements:**

- Each extension entry MUST specify types and constraints for parameters and return values.
- The extension’s internal logic MUST NOT violate any of these constraints.
- Allowed capabilities define the operational scope of the entry, ensuring fine-grained access control.

**Example:**

```yaml
extension_entries:
  - name: "process_data"
    signature:
      params:
        - { type: "int_positive", name: "count" }
      return_type: "int_positive"
    constraints:
      - "return ≥ 0"
      - "memory_usage ≤ 65536"
      - "side_effects = false"
    allowed_capabilities:
      - "FileRead"
```

Here, `process_data` can read files via `host_read_file` but not perform network operations or other disallowed actions.

### 10. Verification Process

Before an extension is integrated and executed, the verifier checks:

1. **Type Checking**: Validate that parameters and returns match the specified types and constraints.
2. **Constraint Compliance**: Verify that all declared constraints (types, helpers, capabilities, and entries) are satisfied.
3. **Capability Matching**: Ensure the extension only invokes helpers from the allowed capabilities.
4. **Resource and Safety Checks**: Confirm that resource limits and safety conditions (e.g., `side_effects = false`) hold.
5. **Failure Handling**: If any requirement fails, the verifier MUST reject the extension, preventing unsafe integration.

These checks may occur offline (pre-deployment), at compile-time, link-time, or dynamically at runtime, depending on the host’s verification strategy.

### 11. Extensibility and Compliance

The EIM specification is designed to be extensible:

- **Customization**: Implementers can define new base types, constraints, and capabilities as their domain requires.
- **Incremental Updates**: The specification can evolve, and tools can be versioned, allowing ongoing refinement of constraints and capabilities.
- **Compliance Documentation**: Hosts and extension authors SHOULD document which version of the EIM specification they follow and what verifiers they use.

### 12. Non-Normative Notes

Although the EIM framework is language-agnostic, developers can integrate it with type-checkers, static analyzers, symbolic verification tools, or specialized eBPF-style verifiers. Capabilities serve as permission sets, while constraints and types form a contract-based system ensuring correctness and safety. The principles align with a least-privilege philosophy, offering a flexible means to balance security, performance, and programmability.

---

## Summary of Changes and Improvements

1. **Generalization Beyond Dynamic Loading**:  
   - **Before**: The original text focused on extensions as dynamically loaded code.  
   - **After**: We now emphasize that extensions can be integrated at various stages (compile-time, link-time, load-time, or runtime), making the model applicable to a wider range of integration scenarios.

2. **Clarified Terminology**:  
   - **Before**: Some wording suggested a narrow scope for application and verification.  
   - **After**: Terms like "Host Environment" and "Verifier" are now more clearly defined, and we’ve removed language that implies purely dynamic loading.

3. **More Explicit Requirements**:  
   - **Before**: Certain requirements were implied but not always explicitly stated.  
   - **After**: Requirements for constraints, types, and helpers are now stated directly, making it clearer how verification must proceed and what conditions must hold.

4. **Added a Section on Extensibility and Compliance**:  
   - **Before**: There was no direct guidance on evolving or extending the specification.  
   - **After**: A new section on extensibility clarifies how the EIM specification can grow over time and how implementers can indicate compliance and versions.

5. **Refined Non-Normative Notes**:  
   - **Before**: The notes were mostly analogies to OOP concepts.  
   - **After**: The notes now emphasize verification techniques and the principle-of-least-privilege design philosophy, offering broader context to potential implementers.

6. **Improved Consistency and Organization**:  
   - **Before**: Some sections mixed normative and descriptive language.  
   - **After**: The text now consistently uses normative language where required, and descriptive examples are clearly delineated.

Through these changes, the revised EIM specification presents a more flexible, clearer, and more general framework that applies to a range of extension-integration models, not just dynamic loading. This updated specification is better structured, more explicit, and easier to adapt to different systems and verification strategies.
