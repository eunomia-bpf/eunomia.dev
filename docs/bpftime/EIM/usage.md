# EIM Usage and people involved (Draft v1)

Below is a description of how various stakeholders in the eBPF extension ecosystem can use and benefit from the Extension Permission and Control Model (EIM). The EIM defines a structured, verifiable way to allow extensions (like eBPF code) to run within a host environment (like a kernel or userspace runtime) securely and reliably.

We have several types of people involved in the ecosystem:

1. **Runtime Developers (eBPF runtime developers)**  
2. **Application Developers**  
3. **Security/Policy Authors (those who define security requirements and models)**  
4. **Extension Developers**  
5. **End Users of Extensions**

The EIM and the concepts of roles, capabilities, attributes, typed interfaces, and formal verification affect each stakeholder differently.

---

### 1. Runtime Developers (eBPF runtime developers)  
**Role:** They build and maintain the underlying system that loads, verifies, and executes eBPF-based extensions.  

**How They Use EIM:**  
- **Defining the Verification Framework:**  
  They use EIM concepts to implement a generic verifier that can enforce typed interfaces, capability rules, and attribute constraints. Their runtime includes logic to:
  - Parse Roles, Capabilities, Attributes, and Types.
  - Perform static or symbolic verification to ensure extension compliance.
  
- **Ensuring Generic and Portable Principles:**  
  They strive to keep runtime principles generic enough to apply across multiple environments. For example:
  - Running on kernel space vs. userspace.
  - Supporting various OS or containerized environments.
  The runtime should be flexible, allowing application developers to define their own sets of host APIs, capabilities, and roles without altering the runtime’s core logic.

- **Scalability to New Scenarios:**  
  Their contribution is making sure the EIM constructs (roles, attributes, capabilities) are sufficiently abstract to handle new host functions or new safety models. This means:
  - The runtime’s verification engine must adapt to new host APIs easily.
  - The runtime must handle evolving sets of attributes (e.g., new memory constraints, new safety flags) without code changes.
  
**Outcome for Runtime Developers:**  
They provide a stable, extensible foundation. Their success is measured by how easily new applications and extensions integrate with minimal adjustments.

---

### 2. Application Developers  
**Role:** They create applications that can be extended by external eBPF code. They define the extension points, which roles are available, and what host APIs can be called.

**How They Use EIM:**  
- **Defining Abstractions for Extension Points:**  
  They identify hook points in their application (like “on request arrival” or “on event X”) and specify which capabilities and roles apply there. For instance:
  - A network application might define a `NetworkObserverRole` that grants read-only access to certain packet data (capability `PacketRead`) and sets attributes like `max_memory`.
  - A monitoring application might have a `TracingRole` that allows calling `bpf_probe_read` and related APIs, but not altering memory.

- **Creating uFuncs and Hook Points:**  
  Application developers use “uFuncs” or extension entry points as a stable API surface for extensions. They map these uFuncs to Roles:
  - E.g., `UPROBE_ENTRY(extension_entry, (int, int) -> void)` might be associated with a `UprobeRole`.
  
- **Abstracting Away Requirements:**  
  They start from what the extension should do (e.g., read some data, perform a calculation) and formalize the required capabilities and attributes. They produce a configuration that the runtime and verifier can easily check.

**Outcome for Application Developers:**  
They offer a clearly defined interface and environment for extensions, ensuring that only safe, verified code runs within their application’s extension points.

---

### 3. Security/Policy Authors (Those Who Define Security Requirements)  
**Role:** They specify security policies, constraints, and safety models for extensions. This may be security officers, system architects, or admins who set global policies on what extensions can do.

**How They Use EIM:**  
- **Defining Roles and Capabilities Aligned with Security Policies:**  
  They write down rules like:  
  - “Extensions in `BasicSecurityRole` can only read memory, not write it.”  
  - “`NetworkSafeRole` must have `network_access = false` to prevent outgoing connections.”
  
- **No Changes to Verifier Code Needed:**  
  They rely on the runtime’s generic verification engine and EIM’s flexible attribute system. They don’t need to rewrite the verifier for every new policy. Instead, they:
  - Add attributes like `may_have_side_effect = false` to forbid certain operations.
  - Compose capabilities to disallow certain host APIs or restrict usage frequency.
  
- **Future Work and Safety Models:**  
  Security authors can experiment with different safety models by adjusting roles and attributes. For example, they can define a “StrictRole” that enforces more stringent constraints. Future expansions could add new attributes or capabilities without modifying the verifier’s code, just the policy definitions.

**Outcome for Security Authors:**  
They gain a systematic way to define and enforce security models, ensuring that any extension admitted into the environment meets their prescribed safety and resource constraints.

---

### 4. Extension Developers  
**Role:** They write the actual extension code (e.g., eBPF programs) that plug into the application’s defined hook points.

**How They Use EIM:**  
- **Adhering to Typed Interfaces and Constraints:**  
  They see a clearly documented set of Roles and Capabilities. For example:
  - If they want to run under `UprobeRole`, they know they can call `bpf_probe_read`. If they call unavailable APIs, the verifier will reject their extension.
  - They must ensure their code never tries to call APIs not allowed in their assigned role.
  
- **Meeting Attribute Constraints:**  
  They must write code that fits within `max_memory` and respects `allow_memory_write`. If their role doesn’t allow memory writes, they must avoid them or risk verification failure.
  
- **Confidence Through Verification:**  
  By following the typed interfaces and constraints, extension developers can be confident that if their extension passes verification, it will run safely in the target environment. They don’t have to guess what’s allowed—they consult the role and capability definitions.

**Outcome for Extension Developers:**  
They have a clear guide on what is allowed and how to write safe, accepted extension code. They focus on logic rather than compatibility guesswork.

---

### 5. End Users of Extensions  
**Role:** They deploy and use these extensions in their environment, benefiting from added functionality.

**How They Use EIM:**  
- **Trust in the System:**  
  When a user picks an extension, they trust that it’s been verified against the roles and capabilities defined by the application and security policies. They don’t need to understand the low-level details.
  
- **Simplicity and Reliability:**  
  They just load the extension. If it passes verification, it runs safely. If it fails verification, the system rejects it, preventing unsafe scenarios.
  
- **Seamless Experience:**  
  Users get a stable system that doesn’t crash or misbehave due to ill-defined extensions because all constraints are checked beforehand.

**Outcome for End Users:**  
They enjoy a more robust, secure experience without dealing with the complexities of verification. They benefit from safe, stable, and predictable extension behavior.

---

## Putting It All Together

- **Runtime Developers** ensure the infrastructure and verification logic can handle flexible sets of rules.
- **Application Developers** define how their applications can be extended, creating roles and capabilities that shape the extension environment.
- **Security/Policy Authors** set the ground rules and security models as attributes and constraints, ensuring compliance with organizational or system-level policies.
- **Extension Developers** use these rules and definitions to write compliant, secure extensions that pass verification.
- **End Users** trust the system to enforce rules, letting them safely run extensions without requiring deep technical involvement.

In summary, the EIM specification creates a structured ecosystem where each stakeholder knows their part: runtime developers build the verification engine, application developers define extension points, security authors specify constraints, extension developers write code within these constraints, and end users run verified extensions confidently.