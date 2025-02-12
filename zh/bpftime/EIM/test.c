#include "interface.h"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

//------------------------------------
// Define Host APIs as UFunctions
//------------------------------------
// Suppose the host provides these file operations and network operations:
int host_open_file(const char* filename, int flags){
    printf("Opening file: %s\n", filename);
    return open(filename, flags);
}
BPFTIME_EXTENSION_UFUNCTION_DEFINE(file_ops, host_open_file)
int host_read_file(int fd, char* buffer, int length){
    printf("Reading file: %d\n", fd);
    return read(fd, buffer, length);
}
BPFTIME_EXTENSION_UFUNCTION_DEFINE(file_ops, host_read_file)
int host_write_file(int fd, const char* buffer, int length){
    printf("Writing file: %d\n", fd);
    return write(fd, buffer, length);
}
BPFTIME_EXTENSION_UFUNCTION_DEFINE(file_ops, host_write_file)

//------------------------------------
// Define Capabilities
//------------------------------------
BPFTIME_CAPABILITY_DEFINE(FileRead)
BPFTIME_CAPABILITY_ADD_UFUNCTION(FileRead, file_ops, host_open_file)
BPFTIME_CAPABILITY_ADD_UFUNCTION(FileRead, file_ops, host_read_file)

BPFTIME_CAPABILITY_DEFINE(FileWrite)
BPFTIME_CAPABILITY_ADD_UFUNCTION(FileWrite, file_ops, host_write_file)

//------------------------------------
// Define Roles and Set Attributes
//------------------------------------
BPFTIME_ROLE_DEFINE(BasicFileRole)
BPFTIME_ROLE_ADD_CAPABILITY(BasicFileRole, FileRead)
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, max_memory, 1048576)      // 1 MB
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, max_instructions, 1000000)
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, allow_memory_write, 0)
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, network_access, 0)
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, always_const, 1)
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, may_have_side_effect, 0)
BPFTIME_ROLE_SET_ATTRIBUTE(BasicFileRole, skip_verification, 0)

BPFTIME_ROLE_DEFINE(ExtendedFileRole)
BPFTIME_ROLE_ADD_CAPABILITY(ExtendedFileRole, FileWrite)
BPFTIME_ROLE_INHERIT(ExtendedFileRole, BasicFileRole)

BPFTIME_ROLE_DEFINE(AdminRole)
BPFTIME_ROLE_INHERIT(AdminRole, ExtendedFileRole)
BPFTIME_ROLE_SET_ATTRIBUTE(AdminRole, allow_memory_write, 1)
BPFTIME_ROLE_SET_ATTRIBUTE(AdminRole, network_access, 1)
BPFTIME_ROLE_SET_ATTRIBUTE(AdminRole, may_have_side_effect, 1)

//------------------------------------
// Define Extension Entries for a Role
//------------------------------------
// Here we associate extension entries with the AdminRole
// No custom logic, only NOPs (from the macro).
BPFTIME_EXTENSION_ENTRY_FOR_ROLE(AdminRole, process_string, const char* input, int mode)
BPFTIME_EXTENSION_ENTRY_FOR_ROLE(AdminRole, analyze_data, const char* data, int length)

int original_process_data(const char* input, int mode) {
    // user notation for using extension to replace original function
    int result = BPFTIME_EXTENSION_ENTRY_INVOKE(AdminRole, process_string, "hello", 42);
    if (result > 0) {
        return result;
    }
    // original logic
    // mock it
    return strlen(input);
}

//------------------------------------
// Demo usage
//------------------------------------
int main() {
    int result = BPFTIME_EXTENSION_ENTRY_INVOKE(AdminRole, process_string, "hello", 42);
    printf("Result: %d\n", result);

    int analyzed = BPFTIME_EXTENSION_ENTRY_INVOKE(AdminRole, analyze_data, "data", 4);
    printf("Analyzed: %d\n", analyzed);

    return 0;
}
