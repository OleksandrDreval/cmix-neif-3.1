// Modified in November 2026 by Oleksandr Dreval (https://github.com/OleksandrDreval)
// for cmix-neif-3.1 (https://github.com/OleksandrDreval/cmix-neif-3.1).
// SPDX-License-Identifier: GPL-3.0-only
// glibc_compat.cpp - keeps the binary on symbol versions every glibc >= 2.34
// provides.
//
// glibc 2.38 headers redirect strtol to __isoc23_strtol under _GNU_SOURCE
// (which C++ always defines); the redirected symbol exists only from glibc
// 2.38 on, so a binary built on Ubuntu 24.04 did not start on Ubuntu 22.04
// ("GLIBC_2.38 not found").  The C23 variant differs only in accepting a
// 0b/0B prefix for base 0 or 2; every call in this program uses base 8 or 10,
// so the classic entry point gives identical results.  This unit defines
// __isoc23_strtol and forwards it to strtol.  It includes no C library header
// on purpose: the headers would apply the redirect here too.

extern "C" {
long compat_strtol(const char* s, char** end, int base) __asm__("strtol");

long __isoc23_strtol(const char* s, char** end, int base) {
  return compat_strtol(s, end, base);
}
}

// The four std::__throw_* the image calls end in a
// trap here instead of being imported from libstdc++.  They are reached only
// on a broken invariant (length beyond max_size, failed allocation); the build
// has no exceptions, so libstdc++ would end in abort as well.  The version
// script keeps them local (no export).
namespace std {
[[noreturn]] void __throw_length_error(const char*) { __builtin_trap(); }
[[noreturn]] void __throw_bad_array_new_length() { __builtin_trap(); }
[[noreturn]] void __throw_bad_alloc() { __builtin_trap(); }
[[noreturn]] void __throw_logic_error(const char*) { __builtin_trap(); }
}  // namespace std

// Plain new/delete straight to malloc/free,
// one indirection and two imports less.  Not in builds whose allocator object
// defines them itself.
extern "C" void* malloc(decltype(sizeof 0));
extern "C" void free(void*);
void* operator new(decltype(sizeof 0) n) { return malloc(n); }
void operator delete(void* p) noexcept { free(p); }
