#ifndef BPFTIME_ATTACH_INTERFACE_H
#define BPFTIME_ATTACH_INTERFACE_H

#if defined(__x86_64__) || defined(__i386__)
    #define INSERT_NOP __asm__ volatile("nop")
#elif defined(__arm__) || defined(__aarch64__)
    #define INSERT_NOP __asm__ volatile("nop")
#else
    #define INSERT_NOP do { } while (0)
#endif

#define INSERT_NOPS(count) \
    do { \
        for (int __i = 0; __i < (count); __i++) { \
            INSERT_NOP; \
        } \
    } while (0)

//------------------------------------
// Host APIs (UFunctions)
//------------------------------------
// Define a host function marker. "name" is a logical category, "function" is the actual host symbol.
#define BPFTIME_EXTENSION_UFUNCTION_DEFINE(name, function) \
    typedef int __bpftime_ext_ufunction_##name##_##function; \
    __bpftime_ext_ufunction_##name##_##function _##function;

//------------------------------------
// Capabilities
//------------------------------------
// A capability represents a set of allowed host APIs.
#define BPFTIME_CAPABILITY_DEFINE(cap_name) \
    typedef int __bpftime_capability_##cap_name;

#define BPFTIME_CAPABILITY_ADD_UFUNCTION(cap_name, ufunc_name, ufunc_symbol) \
    typedef int __bpftime_capability_##cap_name##_includes_##ufunc_name##_##ufunc_symbol; \
    __bpftime_capability_##cap_name##_includes_##ufunc_name##_##ufunc_symbol __bpftime_cap_##cap_name##_##ufunc_name##_##ufunc_symbol;

//------------------------------------
// Attributes
//------------------------------------
// Attributes represent both resource and behavioral constraints.
#define BPFTIME_ATTRIBUTE_DEFINE(attr_name) \
    typedef int __bpftime_attribute_##attr_name;

#define BPFTIME_ATTRIBUTE_VALUE(attr_name, value) \
    enum { __bpftime_attribute_val_##attr_name = (value) };

// Example attributes
BPFTIME_ATTRIBUTE_DEFINE(max_memory)
BPFTIME_ATTRIBUTE_DEFINE(max_instructions)
BPFTIME_ATTRIBUTE_DEFINE(allow_memory_write)
BPFTIME_ATTRIBUTE_DEFINE(network_access)
BPFTIME_ATTRIBUTE_DEFINE(always_const)
BPFTIME_ATTRIBUTE_DEFINE(may_have_side_effect)
BPFTIME_ATTRIBUTE_DEFINE(skip_verification)

//------------------------------------
// Roles
//------------------------------------
// A role aggregates capabilities and attributes, may inherit other roles.
#define BPFTIME_ROLE_DEFINE(role_name) \
    typedef int __bpftime_role_##role_name;

#define BPFTIME_ROLE_ADD_CAPABILITY(role_name, cap_name) \
    typedef int __bpftime_role_##role_name##_has_cap_##cap_name; \
    __bpftime_role_##role_name##_has_cap_##cap_name __bpftime_r_##role_name##_c_##cap_name;

#define BPFTIME_ROLE_SET_ATTRIBUTE(role_name, attr_name, value) \
    enum { __bpftime_role_##role_name##_##attr_name = (value) };

#define BPFTIME_ROLE_INHERIT(role_name, parent_role) \
    typedef int __bpftime_role_##role_name##_inherits_##parent_role; \
    __bpftime_role_##role_name##_inherits_##parent_role __bpftime_r_##role_name##_parent_##parent_role;

//------------------------------------
// Extension Entries for a Role
//------------------------------------
// Extension entry points for a role. No custom logic, just NOPs.
#define BPFTIME_EXTENSION_ENTRY_FOR_ROLE(role, name, ...) \
    int __bpftime_extension_entry_##role##_##name(__VA_ARGS__) { \
        INSERT_NOPS(8); \
        return 0; \
    } \
    typedef int __bpftime_ext_type_##role##_##name; \
    __bpftime_role_##role __bpftime_ext_role_entry_marker_##role##_##name;

#define BPFTIME_EXTENSION_ENTRY_INVOKE(role, name, ...) \
    __bpftime_extension_entry_##role##_##name(__VA_ARGS__)

#endif // BPFTIME_ATTACH_INTERFACE_H
