# EIM: Extension Interface Model Specification

## 1. Overview

**EIM** is a framework for describing _fine-grained_ interactions between extensions (plugins) and a host application. Its central principle is the **principle of least privilege**: each extension should receive only the minimal resources and interactions required to fulfill its task.

EIM addresses two competing needs:

1. **Interconnectedness**: Extensions may require access to specific functions or global state in the host.  
2. **Safety**: The host may want to ensure that buggy extensions can’t crash the system or cause security issues.

### 1.1. Background: Why EIM?

Historically, extension frameworks have faced the problem of letting users do interesting things while not endangering the host. Traditional approaches (like letting all extensions read/write everything in the host) can lead to:

- **Bugs**: A single extension might crash the entire application or cause unexpected side effects.  
- **Security Incidents**: If an extension can corrupt memory or call privileged functions, it can compromise the host.

EIM avoids these pitfalls by **precisely enumerating** host resources as capabilities. This also has benefits for maintenance and auditing:

- **Clear Separation**: The developer defines all possible extension points once.  
- **Reusable**: Administrators can add or remove capabilities in the deployment spec without changing the host’s code.  
- **Auditable**: An EIM specification is a single document that auditors can review to see which resources are actually granted to each extension.

### 1.2. Key Ideas

1. **Resources**: Any “thing” in the host application or the underlying system that an extension might need or want to use—like CPU cycles, memory, global variables, or function calls.  
2. **Capabilities**: A grant of permission to use a given resource (e.g., the right to read a host variable, the right to call a particular function, or the right to consume a certain amount of CPU instructions).  
3. **Extension Entries**: Specific hooks or override points in the host application that can be extended. For instance, the host might allow hooking function `foo()`, or hooking the “request received” phase in a webserver.

An **EIM specification** codifies, in detail, the resources an application can provide (development-time) and how those resources are allocated to each extension entry (deployment-time).

---

## 2. Concepts

This section explains the terms relevant to EIM: **extensions**, **resources**, **capabilities**, and **extension entries**.

### 2.1 Extensions

An **extension** is a piece of software that _plugs into_ a host application at runtime. The host is generally not developed by the extension author (e.g., a user extending nginx, Redis, or some other third-party application). Since direct modifications to the host code can be risky or cumbersome, an extension framework provides safe ways to alter or augment the host’s behavior.

### 2.2 Resources

A **resource** is anything the extension might need to interact with. Typical resources include:

- **Host Variables**: Specific variables (like `ngx_pid` in nginx) or global data structures (e.g., global counters).  
- **Host Functions**: Functions within the host that the extension might call (e.g., `nginxTime()` or `processRequest()`).  
- **Memory / CPU Quotas**: If an extension is allowed to consume CPU cycles or memory, EIM can represent these as numeric resource limits (e.g., `instructions<10000`).  
- **Runtime-Defined Actions**: The extension framework may introduce additional “helper” functions or internal data structures (e.g., “map helpers” in eBPF-based runtimes).

### 2.3 Capabilities

A **capability** is the specific permission to use a given resource. For instance:

- **State Capability**: `readPid` could allow an extension to _read_ a global variable `ngx_pid`.  
- **Function Capability**: `nginxTime` might allow calling a function that returns a timestamp, with constraints like “return value must be positive.”  
- **Resource Capability**: `instructions < 100000` might bound how many instructions an extension can run, or `memory < 1MB` might limit the memory it can allocate.

### 2.4 Extension Entries

An **Extension Entry** is a hook point in the host application. Extensions replace or wrap the behavior of a function or logic block at that entry. For example, if `processRequest` in a webserver is declared as an extension entry, an extension can run in place of (or in tandem with) the default `processRequest` code.

Often, these entries are “function interposition” points, but EIM does not require them to be at function boundaries only. They could also be inserted in a code path (e.g., after some initialization routine).

---

## 3. EIM Specification: Structure

EIM splits its specification into **two** distinct parts:

1. **Development-Time**: Produced by the application developer, who knows how the host is structured.  
2. **Deployment-Time**: Created by a system administrator who decides how to allocate the host’s resources to each potential extension usage.

By separating these steps, developers do _not_ need to foresee every possible extension’s exact safety constraints. They simply enumerate all possible resources (variables, functions, etc.). The system administrator then tailors which subset of those resources each extension may actually use.

### 3.1 Development-Time Specification

**Goal**: Enumerate all the _possible_ resources the host can offer and define extension entries that can be overridden.

At this stage, the developer typically includes:

1. **State Capabilities**  
   - Named permissions to read or write certain host variables.  
   - Form: `State_Capability(name, operation=read|write(variable))`.

2. **Function Capabilities**  
   - Named permissions to call host functions.  
   - Must include the function prototype and any constraints (pre-/post-conditions).  
   - Form:  

     ```text
     Function_Capability(
         name          = some_name,
         prototype     = (arg_types...) -> return_type,
         constraints   = { some_condition_on_args_and_return }
     )
     ```

3. **Extension Entries**  
   - Define the points in the host that can be extended.  
   - Must include the original function name (or similar hook reference) and a prototype.  
   - Form:  

     ```text
     Extension_Entry(
         name              = human_readable_name,
         extension_entry   = actual_host_symbol_name,
         prototype         = (arg_types...) -> return_type
     )
     ```

#### 3.1.1 Example Development-Time Spec

Below is a **development-time** EIM snippet for nginx:

```c
State_Capability(
    name = "readPid",
    operation = read(ngx_pid)
)

Function_Capability(
    name = "nginxTime",
    prototype = (void) -> time_t,
    constraints = {rtn > 0}
)

Extension_Entry(
    name="processBegin",
    extension_entry="ngx_http_process_request",
    prototype = (Request *r) -> int
)

Extension_Entry(
    name="updateResponseContent",
    extension_entry="ngx_http_content_phase",
    prototype = (Request *r) -> int
)
```

- **`readPid`**: Allows an extension to read `ngx_pid`.  
- **`nginxTime`**: Allows calling a timestamp function, with a constraint that its return value (`rtn`) is > 0.  
- **`processBegin`**: A place to insert custom logic right after parsing a request.  
- **`updateResponseContent`**: A place to modify the response’s content before sending it back.

### 3.2 Deployment-Time Specification

**Goal**: Assign actual capabilities (from the development-time specification) to each extension entry. The admin weighs how much interconnectedness each extension should have versus how restricted it should be for safety.

The **main object** at deployment time is an **Extension Class**, which includes:

- A **name**  
- The **extension entry** it targets  
- A set of **capabilities** allowed for that entry

Administrators can also add resource capabilities (e.g., bounding CPU or memory) or specify whether the extension can read or write to certain function arguments or local variables.

#### 3.2.1 Example Deployment-Time Spec

Below is a **deployment-time** EIM snippet for the nginx example above:

```c
Extension_Class(
    name = "observeProcessBegin",
    extension_entry = "processBegin",
    allowed = { instructions<inf, nginxTime, readPid, read(r) }
)

Extension_Class(
    name = "updateResponse",
    extension_entry = "updateResponseContent",
    allowed = { instructions<inf, read(r), write(r) }
)
```

- **`observeProcessBegin`**  
  - Binds to the `processBegin` extension entry.  
  - Allows unlimited instructions (`instructions<inf`), calling `nginxTime`, reading `ngx_pid`, and reading from the argument `(Request* r)`.  
- **`updateResponse`**  
  - Binds to the `updateResponseContent` extension entry.  
  - Grants unlimited instructions, read/write access to `(Request* r)`.

This example shows typical usage where an extension might read the request for logging/analysis, while another extension can modify the request or response. By enumerating them separately, EIM ensures each extension gets only the resources it actually needs.

---

## 4. Detailed Semantics and Rules

EIM is flexible but rests on a few key semantics:

1. **Capability Unions**: An extension entry might require multiple capabilities. The final set of allowed capabilities for an extension is the union of those declared in its **Extension Class**.  
2. **Argument Capabilities**: EIM uses the function prototype in the extension entry. For example, `read(r)` indicates read-access to the pointer argument. If an extension needs to modify it, the admin must grant `write(r)`.  
3. **Constraints**: A function capability can include conditions on arguments or the return value. These constraints are enforced by the extension framework at load time, ensuring that the extension’s code does not violate them (e.g., by preventing calls with invalid arguments).  
4. **Multiple Classes**: An administrator could define multiple **Extension Classes** for the same **Extension Entry** if different extensions require different levels of access. For example, two classes might exist for `processBegin`—one for read-only logging, another for read-write firewall actions.  
5. **Resource Controls**: EIM includes resource capabilities like `instructions<10000` or `memory<2MB`. The specifics of how those are enforced can vary by the runtime (via just-in-time verification or other means).

---

## 5. EIM Usage Workflow

1. **Host Developer**  
   - During development of the host, they annotate or otherwise specify:  
     - Which variables or functions can be extended and how.  
     - Which points in the code are open to extension.  
   - Produces a “development-time spec” listing all possible capabilities and extension entries.

2. **System Administrator**  
   - For each real use-case (e.g., a firewall extension or an observability extension), decides how much access is needed.  
   - Creates one or more **Extension Classes** in the “deployment-time spec.”  
   - Adjusts safety vs. interconnectedness according to site-specific constraints.

3. **Extension Writer**  
   - Writes an extension referencing a particular Extension Class.  
   - The runtime or loader enforces that the extension only uses the capabilities enumerated in that class.

---

## 6. Example in Action

Below is a short hypothetical scenario:

1. **Host Developer**: Writes a video-processing application that can be extended with filters. The developer identifies:  
   - A variable `frameCount` to track frames processed.  
   - A function `applyFilter(Frame *f) -> int`.  
   - Two extension entries: `beforeRender` and `afterRender`.  
   - A function capability `logMessage(char *msg)` to write logs.

2. **Development-Time Spec** might be:

   ```c
   State_Capability(
       name = "readFrameCount",
       operation = read(frameCount)
   )

   Function_Capability(
       name = "logger",
       prototype = (char *) -> void,
       constraints = {}  // No special constraints
   )

   Extension_Entry(
       name = "beforeRenderHook",
       extension_entry = "beforeRender",
       prototype = (Frame *f) -> int
   )

   Extension_Entry(
       name = "afterRenderHook",
       extension_entry = "afterRender",
       prototype = (Frame *f) -> int
   )
   ```

3. **System Administrator**: Decides how to deploy:

   ```c
   Extension_Class(
       name = "loggingExtension",
       extension_entry = "afterRenderHook",
       allowed = { instructions<50000, logger, readFrameCount, read(f) }
   )

   Extension_Class(
       name = "watermarkExtension",
       extension_entry = "afterRenderHook",
       allowed = { instructions<50000, read(f), write(f) }
   )
   ```

4. **Extension Developer**:
   - Writes an eBPF (or other) extension referencing “loggingExtension” if they want to read `frameCount` and log statuses.  
   - Another extension references “watermarkExtension,” which can actually modify the `Frame *f` to embed a watermark.

This separation of concerns is one of the main advantages of EIM. The host developer describes the “universe of possible extension resources,” while the admin determines which portion of that universe each extension gets.

---

## 7. Advanced Features & Notes

1. **Constraints on Function Calls**  EIM constraints can do more than “return value > 0.” The admin/developer can define relationships like “arg1 < arg2,” or semantic markers like “return value must be freed by extension.” Enforcement is typically done by the extension framework’s verifier or at runtime.

2. **Parameter-Level Access**  Some frameworks allow specifying that an extension may read only certain fields of a struct passed as an argument (more fine-grained than just `read(r)`). EIM can represent this, but it depends on the runtime’s ability to enforce.  

3. **Multiple Classes for One Entry**   The same extension entry can appear in multiple extension classes with different capabilities. This is helpful if multiple extensions hooking the same function require different privileges.

4. **Resource Capabilities**   `instructions < X` or `memory < Y` might be enforced either by a static analysis or a runtime boundary. EIM only _declares_ it; how it’s enforced is up to the extension framework (e.g., using an eBPF-style verifier or a hardware-based memory boundary).

5. **Forward Compatibility**   Because the host developer enumerates possible resources, it’s straightforward to add new extension classes later without changing the host code.  

---

## 8. Conclusion

The **Extension Interface Model (EIM)** is a robust way to define _fine-grained_ extension interactions. It puts resource usage under explicit **capabilities** and organizes them into **extension entries**. With EIM, both developers and administrators can confidently add, modify, or remove extensions, knowing each extension can only do what’s explicitly allowed.

### Key Takeaways

- **Least Privilege**: Let each extension do only what it needs.  
- **Clear Separation**: Host developers define the universe of capabilities; admins tailor usage for each extension.  
- **Scalability & Auditing**: EIM specifications can be centrally managed and reviewed.  
- **Flexibility**: Capabilities can represent host states, host functions, or even bounding resources like CPU instructions.

With an appropriate runtime to enforce EIM (e.g., via static verification and hardware-based isolation), this model yields efficient, safe, and maintainable extension deployments.

---

**References** (Partial list from the original text)  

- [27] Orbit  
- [48, 52, 63] Classic references on capabilities  
- [6] Wedge  
- [10, 31, 38] Other frameworks (Shreds, lwC, RLBox)  
- [19] WebAssembly  
- [25] SFI overhead discussions  
- [58, 64] ERIM  
- [67] Native Client (NaCl)

_(The above references are mentioned for historical context. EIM itself is runtime-agnostic—any system that can interpret and enforce the EIM specification is free to do so.)_
