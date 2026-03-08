/*
 * TarsArena — Bump allocator implementation.
 * Uses VirtualAlloc (Windows) or mmap (POSIX) for zero-copy allocation.
 *
 * Agent 1 — Week 1
 */

#include "arena.h"
#include <cstring>
#include <cstdio>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

// ═══════════════════════════════════════
// OS-level memory allocation
// ═══════════════════════════════════════

static void* arena_os_alloc(size_t bytes) {
#ifdef _WIN32
    void* p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) {
        fprintf(stderr, "[TarsArena] VirtualAlloc failed for %zu bytes\n", bytes);
    }
    return p;
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[TarsArena] mmap failed for %zu bytes\n", bytes);
        return nullptr;
    }
    return p;
#endif
}

static void arena_os_free(void* ptr, size_t bytes) {
#ifdef _WIN32
    (void)bytes;
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
#else
    if (ptr) munmap(ptr, bytes);
#endif
}


// ═══════════════════════════════════════
// Arena implementation
// ═══════════════════════════════════════

TarsArena::TarsArena(size_t capacity)
    : base_(nullptr)
    , offset_(0)
    , capacity_(capacity)
    , owns_memory_(true)
{
    base_ = static_cast<uint8_t*>(arena_os_alloc(capacity));
    if (!base_) {
        capacity_ = 0;
    }
}

TarsArena::~TarsArena() {
    if (owns_memory_ && base_) {
        arena_os_free(base_, capacity_);
    }
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
}

void* TarsArena::alloc(size_t bytes, size_t alignment) {
    if (!base_ || bytes == 0) return nullptr;
    
    // Align the current offset up to `alignment`
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + bytes > capacity_) {
        fprintf(stderr, "[TarsArena] Out of memory: requested %zu bytes, "
                "used %zu / %zu\n", bytes, offset_, capacity_);
        return nullptr;
    }
    
    void* ptr = base_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void TarsArena::reset() {
    offset_ = 0;
}
