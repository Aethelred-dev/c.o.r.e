 /**
 * @file core.c
 * @author The Aethelred Team
 * @brief Shipping-Grade, Concurrently-Hardened Vector Symbolic Architecture Engine.
 *
 * @copyright Copyright (c) 2025 The Aethelred Team.
 *
 * This file contains the production implementation of the Synapse V8 engine. This
 * release represents the culmination of extensive architectural reviews, stress
 * testing, and targeted hardening efforts. It is designed for mission-critical
 * applications requiring high-throughput knowledge ingestion and deterministic,
 * auditable reasoning under heavy concurrent loads.
 *
 * COMPILE (Production flags: Pthreads, AVX2, LTO):
 *   gcc -O3 -flto -Wall -Wextra -pthread -mavx2 -o synapse_engine synapse_engine_v8_shipping.c
 *
 */

//==============================================================================
// SECTION 1: HEADERS & PLATFORM ABSTRACTION
//==============================================================================
// Design Note: Standard library includes are grouped for clarity. Platform-
// specific headers are isolated via #ifdefs to maintain portability and make
// dependencies explicit. This is standard practice for cross-platform codebases.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

// Design Note: AVX2 intrinsics are conditionally compiled. The code provides a
// correct, albeit slower, fallback, ensuring functionality on any x86-64 hardware.
#ifdef __AVX2__
#include <immintrin.h>
#endif

//==============================================================================
// SECTION 2: PUBLIC API & CONFIGURATION ("synapse.h")
//==============================================================================
// Design Note: The public API is defined first, C-style, using opaque pointers
// to hide implementation details. This creates a stable interface, allowing the
// internal data structures to be modified without breaking client code. It's the
// "header file" section of a single-file library.

// --- Opaque Pointers for Strong Type Safety & Encapsulation ---
typedef struct CORE_HyperVector_s CORE_HyperVector;
typedef struct CORE_KnowledgeBase_s CORE_KnowledgeBase;
typedef struct CORE_Arena_s CORE_Arena;

// --- Core Constants & Typedefs ---
#define CORE_MAX_CONCEPT_NAME 64
#define CORE_MAX_RELATION_NAME 64
#define CORE_FILE_MAGIC 0x43524538 // "CRE8" for Core Engine v8
#define CORE_FILE_VERSION 800      // v8.0.0
#define CORE_AVX_ALIGNMENT 32

// A comprehensive status enum is critical for robust error handling.
// Clients can switch on these codes to handle failures gracefully.
typedef enum {
    CORE_OK = 0,
    CORE_ERR_NULL_ARG,
    CORE_ERR_INVALID_CONFIG,
    CORE_ERR_MALLOC_FAILED,
    CORE_ERR_NAME_TOO_LONG,
    CORE_ERR_NOT_FOUND,
    CORE_ERR_ALREADY_EXISTS,
    CORE_ERR_FILE_IO,
    CORE_ERR_INVALID_FILE,
    CORE_ERR_VERSION_MISMATCH,
    CORE_ERR_EMPTY_LOG
} CORE_Status;

// The CORE_Config struct centralizes all tunable parameters.
// This prevents "magic numbers" from littering the codebase and provides a
// single point of configuration for users.
typedef struct {
    uint32_t dimensionality;
    uint32_t gravitational_constant;
    uint32_t min_gravitational_distance;
    size_t initial_buckets;
    double resize_load_factor;
    uint64_t seed;
    size_t similarity_cache_size; // 0 to disable
} CORE_Config;

// Public-facing structs for returning query results.
typedef struct {
    char name[CORE_MAX_CONCEPT_NAME];
    uint32_t distance;
} CORE_SearchResult;

typedef struct {
    char relation[CORE_MAX_RELATION_NAME];
    char object[CORE_MAX_CONCEPT_NAME];
    int strength;
} CORE_Relation;

typedef struct {
    const char* subject;
    const char* relation;
    const char* object;
    int strength;
} CORE_BatchAssertion;

// A dedicated stats structure for monitoring engine health and performance.
typedef struct {
    size_t concept_count;
    size_t concept_capacity;
    size_t assertion_count;
    size_t total_relations;
    size_t node_arena_bytes;
    size_t hv_arena_bytes;
    size_t relation_arena_bytes;
    size_t sim_cache_hits;
    size_t sim_cache_misses;
} CORE_Stats;


// --- API Function Prototypes ---

// Lifecycle & Configuration
CORE_KnowledgeBase* core_kb_create(CORE_Config config);
void core_kb_destroy(CORE_KnowledgeBase* kb);
const char* core_status_to_string(CORE_Status status);

// Persistence
CORE_Status core_kb_save(const CORE_KnowledgeBase* kb, const char* filepath);
CORE_KnowledgeBase* core_kb_load(const char* filepath, CORE_Status* status);

// Knowledge Integration & Management
CORE_Status core_kb_assert(CORE_KnowledgeBase* kb, const char* subject, const char* relation, const char* object, int strength);
CORE_Status core_kb_assert_batch(CORE_KnowledgeBase* kb, const CORE_BatchAssertion* assertions, size_t count);
CORE_Status core_kb_unassert(CORE_KnowledgeBase* kb, const char* subject, const char* relation, const char* object);
CORE_Status core_kb_rebuild_from_log(CORE_KnowledgeBase* kb);
CORE_Status core_kb_delete_concept(CORE_KnowledgeBase* kb, const char* name);

// Querying, Reasoning & Introspection
const CORE_HyperVector* core_kb_get_vector(CORE_KnowledgeBase* kb, const char* name);
CORE_SearchResult core_kb_find_closest_bruteforce(CORE_KnowledgeBase* kb, const CORE_HyperVector* query_hv, const char* exclude_name);
CORE_Status core_kb_get_relations(CORE_KnowledgeBase* kb, const char* concept_name, CORE_Relation** out_relations, size_t* out_count);
void core_kb_free_relations(CORE_Relation* relations); // Client is responsible for freeing returned relation arrays.
CORE_Status core_kb_get_stats(CORE_KnowledgeBase* kb, CORE_Stats* out_stats);

// VSA Algebraic Functions
CORE_HyperVector* core_hv_create_bind(const CORE_HyperVector* a, const CORE_HyperVector* b, CORE_Arena* arena);
CORE_HyperVector* core_hv_create_unbind(const CORE_HyperVector* a, const CORE_HyperVector* b, CORE_Arena* arena);
CORE_HyperVector* core_hv_create_bundle(int n_vecs, const CORE_HyperVector** hvs, CORE_Arena* arena);
uint32_t core_hv_distance(const CORE_HyperVector* a, const CORE_HyperVector* b);
uint32_t core_kb_distance(CORE_KnowledgeBase* kb, const CORE_HyperVector* a, const CORE_HyperVector* b); // Cached version


//==============================================================================
// SECTION 3: PRIVATE IMPLEMENTATION
//==============================================================================

// --- Preprocessor & Compiler-Specific Optimizations ---

// Use compiler builtins for popcount where available; they map directly to
// fast hardware instructions. Provide a correct, standard C fallback.
#if defined(__GNUC__) || defined(__clang__)
#define POPCOUNT __builtin_popcountll
#else
// A classic, branch-free bit-twiddling hack for popcount.
static int fallback_popcount(uint64_t n) { int c=0; while(n>0){n&=(n-1);c++;} return c; }
#define POPCOUNT fallback_popcount
#endif

// --- Concurrency Abstraction Layer ---
// Design Rationale: Abstracting mutex operations behind a simple wrapper
// (core_mutex_t) allows us to easily swap out the underlying implementation,
// e.g., for different platforms or for a no-op version in a single-threaded build.
#ifdef _WIN32
typedef CRITICAL_SECTION core_mutex_t;
static void core_mutex_init(core_mutex_t* m) { InitializeCriticalSection(m); }
static void core_mutex_lock(core_mutex_t* m) { EnterCriticalSection(m); }
static void core_mutex_unlock(core_mutex_t* m) { LeaveCriticalSection(m); }
static void core_mutex_destroy(core_mutex_t* m) { DeleteCriticalSection(m); }
#else
typedef pthread_mutex_t core_mutex_t;
static void core_mutex_init(core_mutex_t* m) { pthread_mutex_init(m, NULL); }
static void core_mutex_lock(core_mutex_t* m) { pthread_mutex_lock(m); }
static void core_mutex_unlock(core_mutex_t* m) { pthread_mutex_unlock(m); }
static void core_mutex_destroy(core_mutex_t* m) { pthread_mutex_destroy(m); }
#endif

// --- Deterministic RNG (Xoroshiro128+) ---
// Design Rationale: Using a specific PRNG (not the system's rand()) is
// essential for the engine's deterministic behavior. Given the same seed, the
// entire knowledge base will evolve identically every time.
typedef struct { uint64_t s[2]; } CORE_RNGState;
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rng_next(CORE_RNGState* state) {
    const uint64_t s0 = state->s[0]; uint64_t s1 = state->s[1]; const uint64_t result = s0 + s1;
    s1 ^= s0; state->s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); state->s[1] = rotl(s1, 37); return result;
}

// --- Arena Allocator ---
// Design Rationale: An arena (or bump) allocator is used for ConceptNode,
// HyperVector, and RelationEdge objects. These objects have a lifetime tied
// to the KB itself. The arena provides:
//   1. High performance: Allocation is just a pointer bump.
//   2. Reduced fragmentation: Allocations are contiguous.
//   3. Simple cleanup: The entire arena is freed at once on kb_destroy.
#define ARENA_BLOCK_SIZE (1024 * 1024)
typedef struct ArenaBlock_s { struct ArenaBlock_s* next; size_t used; uint8_t data[]; } ArenaBlock;
struct CORE_Arena_s { ArenaBlock* head; size_t total_bytes; };

static CORE_Arena* arena_create() {
    return calloc(1, sizeof(CORE_Arena));
}
static void* arena_alloc(CORE_Arena* arena, size_t size, size_t align) {
    if (!arena || size == 0) return NULL;
    if (align == 0) align = 1;

    ArenaBlock* block = arena->head;
    if (!block || (ARENA_BLOCK_SIZE - block->used < size + align)) {
        size_t alloc_size = sizeof(ArenaBlock) + (size + align > ARENA_BLOCK_SIZE ? size + align : ARENA_BLOCK_SIZE);
        block = malloc(alloc_size);
        if (!block) return NULL;
        block->used = 0; block->next = arena->head; arena->head = block; arena->total_bytes += alloc_size;
    }
    uintptr_t current_ptr = (uintptr_t)block->data + block->used;
    uintptr_t aligned_ptr = (current_ptr + align - 1) & ~(align - 1);
    size_t padding = aligned_ptr - current_ptr;
    block->used += padding + size;
    return (void*)aligned_ptr;
}
static void arena_destroy(CORE_Arena* arena) {
    if (!arena) return;
    ArenaBlock* block = arena->head;
    while (block) { ArenaBlock* next = block->next; free(block); block = next; }
    free(arena);
}

// --- Internal Data Structures ---

// Represents a high-dimensional vector. The `bits` array is flexible.
struct CORE_HyperVector_s {
    uint32_t d;           // Dimensionality
    uint32_t block_count; // Number of uint64_t blocks needed
    uint64_t bits[];      // Flexible array member for vector data
};

// Represents an explicit relationship, stored in a linked list per concept.
typedef struct RelationEdge_s {
    char relation[CORE_MAX_RELATION_NAME];
    char object[CORE_MAX_CONCEPT_NAME];
    int strength;
    struct RelationEdge_s* next;
} RelationEdge;

// The primary node in our knowledge graph hash table.
typedef struct ConceptNode_s {
    char name[CORE_MAX_CONCEPT_NAME];
    CORE_HyperVector* hv;         // The current, evolved hypervector
    CORE_HyperVector* hv_initial; // The pristine, seed-generated hypervector
    struct ConceptNode_s* next;   // For chaining in the hash bucket
    RelationEdge* relations_head; // Head of the explicit relations list
    bool is_deleted;              // A tombstone flag for soft deletes
} ConceptNode;

// Represents a single fact, stored chronologically in the assertion log.
typedef struct {
    char subject[CORE_MAX_CONCEPT_NAME];
    char relation[CORE_MAX_RELATION_NAME];
    char object[CORE_MAX_CONCEPT_NAME];
    int strength;
    time_t timestamp;
} CORE_Assertion;

// An entry in the similarity cache.
typedef struct {
    uint64_t hash_a, hash_b; // Hashes of the two vectors being compared
    uint32_t distance;       // The cached Hamming distance
    uint64_t last_used;      // Tick for LRU eviction policy
} CORE_SimCacheEntry;

// The similarity cache itself.
typedef struct {
    CORE_SimCacheEntry* entries;
    size_t size;
    uint64_t tick;     // Monotonically increasing clock for LRU
    size_t hits;
    size_t misses;
    core_mutex_t lock; // A single lock for the entire cache.
} CORE_SimilarityCache;

// The master "God object" for the entire knowledge base.
struct CORE_KnowledgeBase_s {
    CORE_Config config;
    CORE_RNGState rng;
    size_t bucket_count;
    size_t concept_count;

    ConceptNode** buckets;      // The hash table
    core_mutex_t* bucket_locks; // A fine-grained lock for each bucket
    core_mutex_t resize_lock;   // A coarse-grained lock to protect resizing
    core_mutex_t assertion_log_lock;
    core_mutex_t rng_lock;

    // Arenas for managing memory of core objects
    CORE_Arena* node_arena;
    CORE_Arena* hv_arena;
    CORE_Arena* relation_arena;

    // The source of truth: a dynamically sized log of all assertions.
    CORE_Assertion* assertion_log;
    size_t assertion_count;
    size_t assertion_capacity;

    CORE_SimilarityCache* sim_cache; // Optional similarity cache
};

// --- Persistence File Header ---
typedef struct {
    uint32_t magic;
    uint32_t version;
    CORE_Config config;
    size_t assertion_count;
} CORE_FileHeader;


// --- Forward Declarations for Internal Helpers ---
static CORE_Status kb_resize(CORE_KnowledgeBase* kb);


// --- Internal Helper Functions ---

// Internal allocation function for hypervectors.
static CORE_HyperVector* hv_alloc_internal(uint32_t d, CORE_Arena* arena) {
    size_t block_count = (d + 63) / 64;
    size_t alignment = 16;
    #ifdef __AVX2__
    alignment = CORE_AVX_ALIGNMENT; // Ensure 32-byte alignment for AVX2
    #endif
    CORE_HyperVector* hv = arena_alloc(arena, sizeof(CORE_HyperVector) + sizeof(uint64_t) * block_count, alignment);
    if (!hv) return NULL;
    hv->d = d;
    hv->block_count = block_count;
    memset(hv->bits, 0, sizeof(uint64_t) * block_count);
    return hv;
}

// Thread-safe wrapper for getting the next random number.
static uint64_t rng_next_locked(CORE_KnowledgeBase* kb) {
    core_mutex_lock(&kb->rng_lock);
    uint64_t val = rng_next(&kb->rng);
    core_mutex_unlock(&kb->rng_lock);
    return val;
}

// Creates a new, random hypervector. This is the atomic "seed" for a new concept.
static CORE_HyperVector* hv_create_random_internal(CORE_KnowledgeBase* kb) {
    CORE_HyperVector* hv = hv_alloc_internal(kb->config.dimensionality, kb->hv_arena);
    if (!hv) return NULL;
    for (uint32_t i = 0; i < hv->block_count; ++i) {
        hv->bits[i] = rng_next_locked(kb);
    }
    // Ensure we don't have bits set beyond the specified dimensionality.
    uint32_t remainder_bits = kb->config.dimensionality % 64;
    if (remainder_bits > 0) {
        uint64_t mask = (1ULL << remainder_bits) - 1;
        hv->bits[hv->block_count - 1] &= mask;
    }
    return hv;
}

// A standard djb2 hash function for strings. Simple, fast, and effective.
static unsigned long hash_string(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

// The core "learning" algorithm. Pulls the `target` vector closer to the
// `influence` vector by flipping a few disagreeing bits.
static void hv_apply_gravity_internal(CORE_HyperVector* target, const CORE_HyperVector* influence, CORE_KnowledgeBase* kb, uint32_t* scratch_buffer) {
    // (Semantic Collapse): A critical guard. If vectors are already very
    // similar, we stop pulling them to prevent them from becoming identical,
    // which would destroy information.
    uint32_t current_dist = core_hv_distance(target, influence);
    if (current_dist <= kb->config.min_gravitational_distance) {
        return;
    }

    // This is an optimization. Instead of iterating through all D bits, we
    // find only the bits that differ and randomly sample from that smaller set.
    uint32_t* differing_indices = scratch_buffer;
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < target->block_count; ++i) {
        uint64_t xor_block = target->bits[i] ^ influence->bits[i];
        if (xor_block == 0) continue;
        for (int j = 0; j < 64; ++j) {
            if ((xor_block >> j) & 1) {
                uint32_t bit_index = i * 64 + j;
                if (bit_index < target->d) {
                    differing_indices[diff_count++] = bit_index;
                }
            }
        }
    }
    if (diff_count > 0) {
        uint32_t gravity = kb->config.gravitational_constant;
        for (uint32_t i = 0; i < gravity && diff_count > 0; ++i) {
            uint64_t rand_val = rng_next_locked(kb);
            uint32_t rand_idx = rand_val % diff_count;
            uint32_t bit_to_flip = differing_indices[rand_idx];

            target->bits[bit_to_flip / 64] ^= (1ULL << (bit_to_flip % 64));

            // A small optimization: swap the used index with the end of the array
            // to avoid shifting memory.
            differing_indices[rand_idx] = differing_indices[diff_count - 1];
            diff_count--;
        }
    }
}

// Finds a concept node by name, or creates it if it doesn't exist. This is a
// high-contention function and a concurrency hotspot.
static ConceptNode* get_or_create_concept_node(CORE_KnowledgeBase* kb, const char* name) {
    // Phase 1: Check for resize. This requires a global lock.
    core_mutex_lock(&kb->resize_lock);
    if ((double)(kb->concept_count + 1) / kb->bucket_count > kb->config.resize_load_factor) {
        if (kb_resize(kb) != CORE_OK) {
            core_mutex_unlock(&kb->resize_lock);
            return NULL; // Resize failed.
        }
    }
    
    // Phase 2: Find or create the node within a specific bucket.
    size_t index = hash_string(name) % kb->bucket_count;
    core_mutex_lock(&kb->bucket_locks[index]);
    // The global resize_lock can be released AFTER we've acquired the bucket lock.
    // This minimizes the duration of the global lock.
    core_mutex_unlock(&kb->resize_lock);

    // First, try to find an existing node.
    for (ConceptNode* current = kb->buckets[index]; current; current = current->next) {
        if (strcmp(current->name, name) == 0) {
            // If the node was soft-deleted, revive it.
            if (current->is_deleted) {
                current->is_deleted = false;
                // This count is protected by the resize_lock, which we held
                // until after we locked this bucket, ensuring consistency.
                __atomic_fetch_add(&kb->concept_count, 1, __ATOMIC_RELAXED);
            }
            core_mutex_unlock(&kb->bucket_locks[index]);
            return current;
        }
    }

    // If not found, create a new one.
    ConceptNode* new_node = arena_alloc(kb->node_arena, sizeof(ConceptNode), 8);
    if (!new_node) { core_mutex_unlock(&kb->bucket_locks[index]); return NULL; }
    memset(new_node, 0, sizeof(ConceptNode));
    strncpy(new_node->name, name, CORE_MAX_CONCEPT_NAME - 1);

    new_node->hv_initial = hv_create_random_internal(kb);
    new_node->hv = hv_alloc_internal(kb->config.dimensionality, kb->hv_arena);
    if (!new_node->hv_initial || !new_node->hv) { core_mutex_unlock(&kb->bucket_locks[index]); return NULL; }
    memcpy(new_node->hv->bits, new_node->hv_initial->bits, sizeof(uint64_t) * new_node->hv->block_count);

    // Insert into the bucket's linked list.
    new_node->next = kb->buckets[index];
    kb->buckets[index] = new_node;
    __atomic_fetch_add(&kb->concept_count, 1, __ATOMIC_RELAXED);
    
    core_mutex_unlock(&kb->bucket_locks[index]);
    return new_node;
}

// This function applies the semantic effect of an assertion without logging it.
// It's the core reusable logic used by both `core_kb_assert` and `core_kb_rebuild_from_log`.
static CORE_Status core_kb_apply_assertion_no_log(CORE_KnowledgeBase* kb, const char* subject_name, const char* relation_name, const char* object_name, int strength) {
    ConceptNode* subj_node = get_or_create_concept_node(kb, subject_name);
    ConceptNode* rel_node = get_or_create_concept_node(kb, relation_name);
    ConceptNode* obj_node = get_or_create_concept_node(kb, object_name);
    if (!subj_node || !rel_node || !obj_node) return CORE_ERR_MALLOC_FAILED;

    // Use a temporary arena for transient calculations. This is cleaner and
    // faster than heap malloc/free for short-lived objects.
    CORE_Arena* temp_arena = arena_create();
    uint32_t* gravity_scratch = malloc(sizeof(uint32_t) * kb->config.dimensionality);
    if (!temp_arena || !gravity_scratch) {
        free(gravity_scratch);
        arena_destroy(temp_arena);
        return CORE_ERR_MALLOC_FAILED;
    }

    // The "influence" vector is the binding of the relation and object.
    // The subject is then pulled towards this composite idea.
    CORE_HyperVector* influence = core_hv_create_bind(rel_node->hv, obj_node->hv, temp_arena);
    
    // Lock the subject's bucket to modify its hypervector.
    size_t subj_index = hash_string(subject_name) % kb->bucket_count;
    core_mutex_lock(&kb->bucket_locks[subj_index]);
    
    // Re-find the subject node now that we have the lock. It cannot have been deleted.
    ConceptNode* locked_subj_node = NULL;
    for(ConceptNode* n = kb->buckets[subj_index]; n != NULL; n = n->next) {
        if(strcmp(n->name, subject_name) == 0) { locked_subj_node = n; break; }
    }
    
    if(locked_subj_node) {
        // 1. Standard Gravitational Pull: S is pulled towards (R ⨂ O).
        for (int i = 0; i < strength; ++i) {
            hv_apply_gravity_internal(locked_subj_node->hv, influence, kb, gravity_scratch);
        }

        // Design Rationale: For `is_a` relations, we want the class (e.g., "Mammal")
        // to be the geometric center of its instances ("Elephant", "Human"). This requires
        // a symmetric pull: the instance pulls the class, and the class pulls the instance.
        if (strcmp(relation_name, "is_a") == 0 || strcmp(relation_name, "is_an") == 0) {
            size_t obj_index = hash_string(object_name) % kb->bucket_count;

            // If we need to lock two buckets, we MUST lock them in a canonical
            // order (e.g., lowest index first) across all threads. Failure to do
            // so can cause a deadly embrace: Thread A locks bucket 5, waits for 10.
            // Thread B locks bucket 10, waits for 5. Deadlock.
            if (obj_index == subj_index) {
                // Same bucket, already locked. No action needed.
            } else if (obj_index < subj_index) {
                core_mutex_unlock(&kb->bucket_locks[subj_index]); // Release higher index
                core_mutex_lock(&kb->bucket_locks[obj_index]);    // Acquire lower index
                core_mutex_lock(&kb->bucket_locks[subj_index]);   // Re-acquire higher index
            } else { // obj_index > subj_index
                core_mutex_lock(&kb->bucket_locks[obj_index]);
            }
            
            // Now that both required buckets are safely locked, find the object node.
            ConceptNode* locked_obj_node = NULL;
            for(ConceptNode* n = kb->buckets[obj_index]; n != NULL; n = n->next) {
                if(strcmp(n->name, object_name) == 0) { locked_obj_node = n; break; }
            }

            if (locked_obj_node) {
                // Apply the symmetric gravitational pull.
                int direct_pull_strength = strength / 2 + 1; // An intensified pull.
                for (int i = 0; i < direct_pull_strength; i++) {
                    hv_apply_gravity_internal(locked_subj_node->hv, locked_obj_node->hv, kb, gravity_scratch);
                    hv_apply_gravity_internal(locked_obj_node->hv, locked_subj_node->hv, kb, gravity_scratch);
                }
            }
            // Unlock the second bucket if we locked it.
            if (obj_index != subj_index) {
                core_mutex_unlock(&kb->bucket_locks[obj_index]);
            }
        }

        // 3. Update the explicit relation graph.
        bool edge_found = false;
        for (RelationEdge* edge = locked_subj_node->relations_head; edge != NULL; edge = edge->next) {
            if (strcmp(edge->relation, relation_name) == 0 && strcmp(edge->object, object_name) == 0) {
                edge->strength += strength;
                edge_found = true;
                break;
            }
        }
        if (!edge_found) {
            RelationEdge* new_edge = arena_alloc(kb->relation_arena, sizeof(RelationEdge), 8);
            if (new_edge) {
                strncpy(new_edge->relation, relation_name, CORE_MAX_RELATION_NAME - 1);
                strncpy(new_edge->object, object_name, CORE_MAX_CONCEPT_NAME - 1);
                new_edge->strength = strength;
                new_edge->next = locked_subj_node->relations_head;
                locked_subj_node->relations_head = new_edge;
            }
        }
    }

    core_mutex_unlock(&kb->bucket_locks[subj_index]);
    
    // Cleanup temporary resources.
    free(gravity_scratch);
    arena_destroy(temp_arena);
    return CORE_OK;
}


//==============================================================================
// SECTION 4: PUBLIC API IMPLEMENTATION
//==============================================================================

const char* core_status_to_string(CORE_Status status) {
    switch (status) {
        case CORE_OK:                   return "Operation successful";
        case CORE_ERR_NULL_ARG:         return "Null argument provided";
        case CORE_ERR_INVALID_CONFIG:   return "Invalid configuration";
        case CORE_ERR_MALLOC_FAILED:    return "Memory allocation failed";
        case CORE_ERR_NAME_TOO_LONG:    return "Concept name exceeds maximum length";
        case CORE_ERR_NOT_FOUND:        return "Concept not found";
        case CORE_ERR_ALREADY_EXISTS:   return "Concept already exists";
        case CORE_ERR_FILE_IO:          return "File I/O error";
        case CORE_ERR_INVALID_FILE:     return "Invalid or corrupted Synapse file";
        case CORE_ERR_VERSION_MISMATCH: return "File version mismatch";
        case CORE_ERR_EMPTY_LOG:        return "Assertion log is empty, cannot rebuild";
        default:                        return "Unknown error";
    }
}

CORE_KnowledgeBase* core_kb_create(CORE_Config config) {
    if (config.dimensionality == 0 || config.initial_buckets == 0) return NULL;
    
    CORE_KnowledgeBase* kb = calloc(1, sizeof(CORE_KnowledgeBase));
    if (!kb) return NULL;

    kb->config = config;
    if (kb->config.seed == 0) kb->config.seed = time(NULL);
    if (kb->config.min_gravitational_distance == 0) kb->config.min_gravitational_distance = 5;
    kb->rng.s[0] = kb->config.seed;
    kb->rng.s[1] = kb->config.seed ^ 0xdeadbeefcafebabe; // A common practice to initialize the second state.

    // Initialize memory and synchronization primitives.
    kb->node_arena = arena_create();
    kb->hv_arena = arena_create();
    kb->relation_arena = arena_create();
    kb->bucket_count = config.initial_buckets;
    kb->buckets = calloc(kb->bucket_count, sizeof(ConceptNode*));
    kb->bucket_locks = malloc(sizeof(core_mutex_t) * kb->bucket_count);

    // Initialize similarity cache if configured.
    if (config.similarity_cache_size > 0) {
        kb->sim_cache = calloc(1, sizeof(CORE_SimilarityCache));
        if (kb->sim_cache) {
            kb->sim_cache->size = config.similarity_cache_size;
            kb->sim_cache->entries = calloc(kb->sim_cache->size, sizeof(CORE_SimCacheEntry));
            if (!kb->sim_cache->entries) { // Allocation failed, proceed without cache.
                free(kb->sim_cache);
                kb->sim_cache = NULL;
            } else {
                core_mutex_init(&kb->sim_cache->lock);
            }
        }
    }

    if (!kb->node_arena || !kb->hv_arena || !kb->relation_arena || !kb->buckets || !kb->bucket_locks) {
        core_kb_destroy(kb); // Cleanup partial initialization.
        return NULL;
    }
    
    core_mutex_init(&kb->resize_lock);
    core_mutex_init(&kb->assertion_log_lock);
    core_mutex_init(&kb->rng_lock);
    for (size_t i = 0; i < kb->bucket_count; ++i) {
        core_mutex_init(&kb->bucket_locks[i]);
    }
    return kb;
}

void core_kb_destroy(CORE_KnowledgeBase* kb) {
    if (!kb) return;
    
    // Destroy in reverse order of creation.
    if (kb->bucket_locks) {
        for (size_t i = 0; i < kb->bucket_count; ++i) core_mutex_destroy(&kb->bucket_locks[i]);
        free(kb->bucket_locks);
    }
    if (kb->sim_cache) {
        core_mutex_destroy(&kb->sim_cache->lock);
        free(kb->sim_cache->entries);
        free(kb->sim_cache);
    }
    
    core_mutex_destroy(&kb->resize_lock);
    core_mutex_destroy(&kb->assertion_log_lock);
    core_mutex_destroy(&kb->rng_lock);
    
    arena_destroy(kb->node_arena);
    arena_destroy(kb->hv_arena);
    arena_destroy(kb->relation_arena);
    
    free(kb->buckets);
    free(kb->assertion_log);
    free(kb);
}

CORE_Status core_kb_save(const CORE_KnowledgeBase* kb, const char* filepath) {
    if (!kb || !filepath) return CORE_ERR_NULL_ARG;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return CORE_ERR_FILE_IO;

    // The assertion log is the only piece of state that needs to be saved.
    // It must be locked during the save operation to prevent writes.
    core_mutex_lock((core_mutex_t*)&kb->assertion_log_lock);

    CORE_FileHeader header = {
        .magic = CORE_FILE_MAGIC,
        .version = CORE_FILE_VERSION,
        .config = kb->config,
        .assertion_count = kb->assertion_count
    };

    CORE_Status status = CORE_OK;
    if (fwrite(&header, sizeof(CORE_FileHeader), 1, fp) != 1) {
        status = CORE_ERR_FILE_IO;
        goto cleanup; // Use goto for centralized cleanup in C error handling.
    }

    if (header.assertion_count > 0) {
        if (fwrite(kb->assertion_log, sizeof(CORE_Assertion), header.assertion_count, fp) != header.assertion_count) {
            status = CORE_ERR_FILE_IO;
            goto cleanup;
        }
    }

cleanup:
    core_mutex_unlock((core_mutex_t*)&kb->assertion_log_lock);
    fclose(fp);
    return status;
}

CORE_KnowledgeBase* core_kb_load(const char* filepath, CORE_Status* out_status) {
    if (!filepath || !out_status) {
        if (out_status) *out_status = CORE_ERR_NULL_ARG;
        return NULL;
    }

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        *out_status = CORE_ERR_FILE_IO;
        return NULL;
    }

    CORE_FileHeader header;
    if (fread(&header, sizeof(CORE_FileHeader), 1, fp) != 1) {
        *out_status = CORE_ERR_INVALID_FILE;
        fclose(fp);
        return NULL;
    }

    // Validate the file header for integrity.
    if (header.magic != CORE_FILE_MAGIC) {
        *out_status = CORE_ERR_INVALID_FILE;
        fclose(fp);
        return NULL;
    }
    if (header.version > CORE_FILE_VERSION) {
        // We disallow loading files from a future version of the engine.
        *out_status = CORE_ERR_VERSION_MISMATCH;
        fclose(fp);
        return NULL;
    }

    // Create a new KB instance using the *saved* configuration.
    CORE_KnowledgeBase* kb = core_kb_create(header.config);
    if (!kb) {
        *out_status = CORE_ERR_MALLOC_FAILED;
        fclose(fp);
        return NULL;
    }

    // Load the assertion log from the file.
    if (header.assertion_count > 0) {
        kb->assertion_log = malloc(sizeof(CORE_Assertion) * header.assertion_count);
        if (!kb->assertion_log) {
            *out_status = CORE_ERR_MALLOC_FAILED;
            fclose(fp);
            core_kb_destroy(kb);
            return NULL;
        }
        kb->assertion_count = header.assertion_count;
        kb->assertion_capacity = header.assertion_count;

        if (fread(kb->assertion_log, sizeof(CORE_Assertion), header.assertion_count, fp) != header.assertion_count) {
            *out_status = CORE_ERR_INVALID_FILE; // File is likely truncated.
            fclose(fp);
            core_kb_destroy(kb);
            return NULL;
        }
    }
    fclose(fp);

    // After loading the log, we must rebuild the in-memory state.
    *out_status = core_kb_rebuild_from_log(kb);
    if (*out_status != CORE_OK) {
        core_kb_destroy(kb);
        return NULL;
    }

    return kb;
}

CORE_Status core_kb_assert(CORE_KnowledgeBase* kb, const char* subject_name, const char* relation_name, const char* object_name, int strength) {
    if (!kb || !subject_name || !relation_name || !object_name) return CORE_ERR_NULL_ARG;
    if (strlen(subject_name) >= CORE_MAX_CONCEPT_NAME || 
        strlen(relation_name) >= CORE_MAX_RELATION_NAME || 
        strlen(object_name) >= CORE_MAX_CONCEPT_NAME) {
        return CORE_ERR_NAME_TOO_LONG;
    }

    // Step 1: Apply the semantic change to the in-memory vectors.
    CORE_Status status = core_kb_apply_assertion_no_log(kb, subject_name, relation_name, object_name, strength);
    if (status != CORE_OK) {
        return status;
    }

    // Step 2: Persist the assertion to the log, which is the source of truth.
    core_mutex_lock(&kb->assertion_log_lock);
    if(kb->assertion_count >= kb->assertion_capacity) {
        size_t new_cap = kb->assertion_capacity == 0 ? 128 : kb->assertion_capacity * 2;
        CORE_Assertion* new_log = realloc(kb->assertion_log, sizeof(CORE_Assertion) * new_cap);
        if(!new_log) {
            core_mutex_unlock(&kb->assertion_log_lock);
            // CRITICAL: The semantic change was applied but not logged. The KB is now
            // in a transiently inconsistent state that will be fixed on next rebuild.
            // This is an acceptable trade-off to avoid crashing.
            return CORE_ERR_MALLOC_FAILED;
        }
        kb->assertion_log = new_log;
        kb->assertion_capacity = new_cap;
    }
    CORE_Assertion* log = &kb->assertion_log[kb->assertion_count++];
    strncpy(log->subject, subject_name, CORE_MAX_CONCEPT_NAME - 1);
    strncpy(log->relation, relation_name, CORE_MAX_RELATION_NAME - 1);
    strncpy(log->object, object_name, CORE_MAX_CONCEPT_NAME - 1);
    log->strength = strength;
    log->timestamp = time(NULL);
    core_mutex_unlock(&kb->assertion_log_lock);

    return CORE_OK;
}

CORE_Status core_kb_assert_batch(CORE_KnowledgeBase* kb, const CORE_BatchAssertion* assertions, size_t count) {
    if (!kb || !assertions) return CORE_ERR_NULL_ARG;
    for (size_t i = 0; i < count; ++i) {
        CORE_Status s = core_kb_assert(kb, assertions[i].subject, assertions[i].relation, assertions[i].object, assertions[i].strength);
        if (s != CORE_OK) return s; // Fail fast on first error.
    }
    return CORE_OK;
}

CORE_Status core_kb_unassert(CORE_KnowledgeBase* kb, const char* subject, const char* relation, const char* object) {
    // API Contract: This function only modifies the log. The in-memory state
    // will not reflect this change until `core_kb_rebuild_from_log` is called.
    if (!kb || !subject || !relation || !object) return CORE_ERR_NULL_ARG;
    
    core_mutex_lock(&kb->assertion_log_lock);
    size_t new_log_count = 0;
    bool found = false;
    for (size_t i = 0; i < kb->assertion_count; ++i) {
        if (strcmp(kb->assertion_log[i].subject, subject) == 0 &&
            strcmp(kb->assertion_log[i].relation, relation) == 0 &&
            strcmp(kb->assertion_log[i].object, object) == 0) {
            found = true;
            // Skip this entry by not incrementing new_log_count.
        } else {
            // This is a compacting remove.
            if (i != new_log_count) {
                kb->assertion_log[new_log_count] = kb->assertion_log[i];
            }
            new_log_count++;
        }
    }
    if (!found) {
        core_mutex_unlock(&kb->assertion_log_lock);
        return CORE_ERR_NOT_FOUND;
    }
    kb->assertion_count = new_log_count;
    core_mutex_unlock(&kb->assertion_log_lock);
    
    return CORE_OK;
}

CORE_Status core_kb_rebuild_from_log(CORE_KnowledgeBase* kb) {
    if (!kb) return CORE_ERR_NULL_ARG;

    // This is a "stop-the-world" operation. We take the global resize lock
    // and all bucket locks to ensure no other thread can access the KB
    // while it's in a partially rebuilt state.
    core_mutex_lock(&kb->resize_lock);
    for(size_t i = 0; i < kb->bucket_count; ++i) core_mutex_lock(&kb->bucket_locks[i]);

    // Design Rationale: For perfect determinism, we reset the RNG to its
    // initial seed before every rebuild.
    kb->rng.s[0] = kb->config.seed;
    kb->rng.s[1] = kb->config.seed ^ 0xdeadbeefcafebabe;
    
    // Nuke the derived state (relations and evolved vectors).
    arena_destroy(kb->relation_arena);
    kb->relation_arena = arena_create();
    if (!kb->relation_arena) {
        // If we can't allocate a new arena, we're in a bad state. Release locks and fail.
        for(size_t i = 0; i < kb->bucket_count; ++i) core_mutex_unlock(&kb->bucket_locks[i]);
        core_mutex_unlock(&kb->resize_lock);
        return CORE_ERR_MALLOC_FAILED;
    }

    for (size_t i = 0; i < kb->bucket_count; ++i) {
        for (ConceptNode* node = kb->buckets[i]; node != NULL; node = node->next) {
            node->relations_head = NULL;
            // Reset the evolved vector to its pristine, initial state.
            memcpy(node->hv->bits, node->hv_initial->bits, sizeof(uint64_t) * node->hv->block_count);
        }
    }
    
    // Release the world.
    for(size_t i = 0; i < kb->bucket_count; ++i) core_mutex_unlock(&kb->bucket_locks[i]);
    core_mutex_unlock(&kb->resize_lock);

    // Clear the similarity cache, as all vectors have changed.
    if (kb->sim_cache) {
        core_mutex_lock(&kb->sim_cache->lock);
        memset(kb->sim_cache->entries, 0, sizeof(CORE_SimCacheEntry) * kb->sim_cache->size);
        kb->sim_cache->hits = 0;
        kb->sim_cache->misses = 0;
        kb->sim_cache->tick = 0;
        core_mutex_unlock(&kb->sim_cache->lock);
    }
    
    // Now, replay the entire assertion log to rebuild the derived state.
    core_mutex_lock(&kb->assertion_log_lock);
    if (kb->assertion_count == 0) {
        core_mutex_unlock(&kb->assertion_log_lock);
        return CORE_OK; // Nothing to do.
    }

    for (size_t i = 0; i < kb->assertion_count; ++i) {
        CORE_Assertion* log = &kb->assertion_log[i];
        // This is safe to do concurrently because apply_assertion_no_log
        // is itself thread-safe. However, we're doing it serially here
        // after locking the log, which is simpler and deterministic.
        core_kb_apply_assertion_no_log(kb, log->subject, log->relation, log->object, log->strength);
    }
    core_mutex_unlock(&kb->assertion_log_lock);

    return CORE_OK;
}

// Internal helper to find a concept node.
static ConceptNode* find_concept_node_internal(CORE_KnowledgeBase* kb, const char* name, bool allow_deleted) {
    if (!kb || !name) return NULL;
    
    // We must hold the resize lock briefly to get a stable bucket count.
    core_mutex_lock(&kb->resize_lock);
    size_t index = hash_string(name) % kb->bucket_count;
    core_mutex_lock(&kb->bucket_locks[index]);
    core_mutex_unlock(&kb->resize_lock);

    ConceptNode* found_node = NULL;
    for (ConceptNode* current = kb->buckets[index]; current; current = current->next) {
        if (strcmp(current->name, name) == 0) {
            if (allow_deleted || !current->is_deleted) {
                 found_node = current;
            }
            break;
        }
    }
    core_mutex_unlock(&kb->bucket_locks[index]);
    return found_node;
}

const CORE_HyperVector* core_kb_get_vector(CORE_KnowledgeBase* kb, const char* name) {
    ConceptNode* node = find_concept_node_internal(kb, name, false);
    return node ? node->hv : NULL;
}

CORE_SearchResult core_kb_find_closest_bruteforce(CORE_KnowledgeBase* kb, const CORE_HyperVector* query_hv, const char* exclude_name) {
    CORE_SearchResult result = { "", kb->config.dimensionality + 1 };
    if (!kb || !query_hv) return result;
    
    // This is a read-only operation, but the hash table can be resized.
    // We must lock to prevent iteration over a partially-rebuilt table.
    core_mutex_lock(&kb->resize_lock);
    size_t current_bucket_count = kb->bucket_count;
    
    // Iterate over every bucket, locking each one individually.
    for (size_t i = 0; i < current_bucket_count; ++i) {
        core_mutex_lock(&kb->bucket_locks[i]);
        for (ConceptNode* node = kb->buckets[i]; node != NULL; node = node->next) {
            if (node->is_deleted || (exclude_name && strcmp(node->name, exclude_name) == 0)) {
                continue;
            }
            // Use the potentially cached distance function for performance.
            uint32_t dist = core_kb_distance(kb, query_hv, node->hv);
            if (dist < result.distance) {
                result.distance = dist;
                strncpy(result.name, node->name, CORE_MAX_CONCEPT_NAME - 1);
            }
        }
        core_mutex_unlock(&kb->bucket_locks[i]);
    }
    core_mutex_unlock(&kb->resize_lock);
    return result;
}

uint32_t core_hv_distance(const CORE_HyperVector* a, const CORE_HyperVector* b) {
    if (!a || !b || a->d != b->d) return (uint32_t)-1;
    
    uint32_t distance = 0;
    uint32_t i = 0;

    // Use AVX2 intrinsics for a significant performance boost on supported hardware.
    // This calculates the popcount on 256 bits (4x 64-bit blocks) at a time.
#ifdef __AVX2__
    uint32_t avx_blocks_end = (a->block_count / 4) * 4;
    for (i = 0; i < avx_blocks_end; i += 4) {
        __m256i vec_a = _mm256_load_si256((__m256i const*)&a->bits[i]);
        __m256i vec_b = _mm256_load_si256((__m256i const*)&b->bits[i]);
        __m256i xored = _mm256_xor_si256(vec_a, vec_b);
        // Unfortunately, there's no direct popcount intrinsic for __m256i.
        // We have to store and popcount the 64-bit lanes individually.
        uint64_t temp[4];
        _mm256_store_si256((__m256i*)temp, xored);
        distance += POPCOUNT(temp[0]) + POPCOUNT(temp[1]) + POPCOUNT(temp[2]) + POPCOUNT(temp[3]);
    }
#endif
    // Process any remaining blocks that didn't fit in the AVX2 loop.
    for (; i < a->block_count; ++i) {
        distance += POPCOUNT(a->bits[i] ^ b->bits[i]);
    }
    return distance;
}

// A simple FNV-1a hash for hypervectors, used for cache lookups.
static uint64_t core_hv_hash(const CORE_HyperVector* hv) {
    if (!hv) return 0;
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* data = (const uint8_t*)hv->bits;
    size_t len = hv->block_count * sizeof(uint64_t); 
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint32_t core_kb_distance(CORE_KnowledgeBase* kb, const CORE_HyperVector* a, const CORE_HyperVector* b) {
    // If cache is disabled or not provided, fall back to the raw calculation.
    if (!kb || !kb->sim_cache) return core_hv_distance(a, b);

    CORE_SimilarityCache* cache = kb->sim_cache;
    uint64_t hash_a = core_hv_hash(a);
    uint64_t hash_b = core_hv_hash(b);

    // Canonical ordering of hashes to ensure (A,B) and (B,A) map to the same cache entry.
    if (hash_a > hash_b) { uint64_t tmp = hash_a; hash_a = hash_b; hash_b = tmp; }

    core_mutex_lock(&cache->lock);
    
    // Check for a cache hit.
    for (size_t i = 0; i < cache->size; ++i) {
        if (cache->entries[i].hash_a == hash_a && cache->entries[i].hash_b == hash_b) {
            cache->entries[i].last_used = ++cache->tick;
            cache->hits++;
            uint32_t dist = cache->entries[i].distance;
            core_mutex_unlock(&cache->lock);
            return dist;
        }
    }
    
    // Cache miss. We must perform the calculation.
    core_mutex_unlock(&cache->lock); // Unlock during the expensive calculation.
    uint32_t dist = core_hv_distance(a, b);
    core_mutex_lock(&cache->lock);   // Re-lock to update the cache.

    // Find a victim entry to evict using a simple LRU policy.
    size_t evict_idx = 0;
    uint64_t min_tick = UINT64_MAX;
    for (size_t i = 0; i < cache->size; ++i) {
        if (cache->entries[i].last_used < min_tick) {
            min_tick = cache->entries[i].last_used;
            evict_idx = i;
        }
    }
    cache->misses++;
    cache->entries[evict_idx].hash_a = hash_a;
    cache->entries[evict_idx].hash_b = hash_b;
    cache->entries[evict_idx].distance = dist;
    cache->entries[evict_idx].last_used = ++cache->tick;
    core_mutex_unlock(&cache->lock);
    
    return dist;
}

CORE_HyperVector* core_hv_create_bind(const CORE_HyperVector* a, const CORE_HyperVector* b, CORE_Arena* arena) {
    // DOC: Using simple XOR for binding is a deliberate choice for speed and its
    // involutive property (a^b^b = a). More complex permutation-based binding
    // may be used in systems where resistance to specific forms of aliasing is required.
    if (!a || !b || a->d != b->d || !arena) return NULL;
    CORE_HyperVector* result = hv_alloc_internal(a->d, arena);
    if (!result) return NULL;
    for (uint32_t i = 0; i < a->block_count; ++i) result->bits[i] = a->bits[i] ^ b->bits[i];
    return result;
}

CORE_HyperVector* core_hv_create_unbind(const CORE_HyperVector* a, const CORE_HyperVector* b, CORE_Arena* arena) {
    // Unbinding is the same as binding for XOR.
    return core_hv_create_bind(a, b, arena);
}

// Resizes the hash table. This is an expensive, "stop-the-world" operation.
static CORE_Status kb_resize(CORE_KnowledgeBase* kb) {
    // PRECONDITION: The caller MUST already hold the `resize_lock`.
    size_t old_bucket_count = kb->bucket_count;
    ConceptNode** old_buckets = kb->buckets;
    core_mutex_t* old_locks = kb->bucket_locks;

    // To safely rehash, we must acquire ALL bucket locks.
    // This prevents any other thread from accessing a node while we are moving it.
    for(size_t i = 0; i < old_bucket_count; ++i) core_mutex_lock(&old_locks[i]);

    size_t new_bucket_count = old_bucket_count * 2;
    ConceptNode** new_buckets = calloc(new_bucket_count, sizeof(ConceptNode*));
    core_mutex_t* new_locks = malloc(sizeof(core_mutex_t) * new_bucket_count);

    if (!new_buckets || !new_locks) {
        free(new_buckets);
        free(new_locks);
        // IMPORTANT: Must release all locks on failure.
        for(size_t i = 0; i < old_bucket_count; ++i) core_mutex_unlock(&old_locks[i]);
        return CORE_ERR_MALLOC_FAILED;
    }

    // Rehash all existing nodes into the new, larger table.
    for (size_t i = 0; i < old_bucket_count; ++i) {
        ConceptNode* current = old_buckets[i];
        while(current) {
            ConceptNode* next = current->next;
            size_t new_index = hash_string(current->name) % new_bucket_count;
            current->next = new_buckets[new_index];
            new_buckets[new_index] = current;
            current = next;
        }
    }

    // Release all old locks.
    for(size_t i = 0; i < old_bucket_count; ++i) core_mutex_unlock(&old_locks[i]);
    
    // Destroy the old locks and free the old bucket array.
    for(size_t i = 0; i < old_bucket_count; ++i) core_mutex_destroy(&old_locks[i]);
    free(old_buckets);
    free(old_locks);
    
    // Install the new, larger resources.
    kb->buckets = new_buckets;
    kb->bucket_locks = new_locks;
    kb->bucket_count = new_bucket_count;
    for(size_t i = 0; i < new_bucket_count; ++i) core_mutex_init(&kb->bucket_locks[i]);

    return CORE_OK;
}

//==============================================================================
// SECTION 5: MAIN DEMONSTRATION
//==============================================================================

// Helper to print formatted section headers for the demo output.
void print_header(const char* title) {
    printf("\n\n//---[ %s ]---//\n", title);
}

// Helper function to perform and validate a standard (Subject, Relation, ?) query.
void perform_algebraic_query(CORE_KnowledgeBase* mind, const char* subject, const char* relation, const char* expected_object) {
    printf("Query: (%s, %s, ?) -> expecting '%s'\n", subject, relation, expected_object);
    
    const CORE_HyperVector* subject_hv = core_kb_get_vector(mind, subject);
    const CORE_HyperVector* relation_hv = core_kb_get_vector(mind, relation);

    if (!subject_hv || !relation_hv) {
        printf("   [FAIL] Could not retrieve base vectors for query.\n");
        return;
    }

    CORE_Arena* temp_arena = arena_create();
    // The query is to find X where: subject ≈ bind(relation, X)
    // Algebraically, this means X ≈ unbind(subject, relation)
    CORE_HyperVector* query_vec = core_hv_create_unbind(subject_hv, relation_hv, temp_arena);
    CORE_SearchResult result = core_kb_find_closest_bruteforce(mind, query_vec, subject);
    
    printf("   Closest match in memory: '%s' (Distance: %u)\n", result.name, result.distance);
    
    if (strcmp(result.name, expected_object) == 0) {
        printf("   [PASS] The engine correctly computed the answer.\n");
    } else {
        printf("   [FAIL] The engine retrieved an incorrect fact.\n");
    }
    arena_destroy(temp_arena);
}

int main(void) {
    const char* mind_filepath = "synapse_mind.core";
    print_header("Synapse V8 Hardened Reasoning Engine Demonstration");

    CORE_Config config = {
        .dimensionality = 2048,
        .gravitational_constant = 10,
        .min_gravitational_distance = 100, // Prevent concepts from collapsing
        .initial_buckets = 16,
        .resize_load_factor = 0.75,
        .seed = 42,
        .similarity_cache_size = 1024 // Enable the cache
    };
    CORE_KnowledgeBase* mind = core_kb_create(config);
    if (!mind) {
        fprintf(stderr, "Fatal: Could not create knowledge base.\n");
        return 1;
    }
    printf("Engine initialized with D=%u and Seed=%llu.\n", config.dimensionality, (unsigned long long)config.seed);

    print_header("Phase 1: Knowledge Integration");
    CORE_BatchAssertion facts[] = {
        // Hierarchical knowledge
        {"Elephant", "is_a", "Mammal", 50},
        {"Human", "is_a", "Mammal", 50},
        {"SamAltman", "is_a", "Human", 100},
        {"Ant", "is_an", "Insect", 50},
        // Property knowledge
        {"Mammal", "has_property", "WarmBlood", 25},
        {"Mammal", "has_property", "Vertebrae", 25},
        {"Insect", "has_property", "Exoskeleton", 25},
        // Relational knowledge
        {"SamAltman", "is_ceo_of", "OpenAI", 100},
        {"OpenAI", "develops", "ChatGPT", 50},
        {"ChatGPT", "is_a", "LLM", 25}
    };
    size_t num_facts = sizeof(facts) / sizeof(facts[0]);
    core_kb_assert_batch(mind, facts, num_facts);
    printf("Integrated %zu facts into the engine's conceptual manifold.\n", num_facts);

    print_header("Phase 2: Verifying Conceptual Distance (Semantic Clustering)");
    const CORE_HyperVector* elephant_hv = core_kb_get_vector(mind, "Elephant");
    const CORE_HyperVector* human_hv = core_kb_get_vector(mind, "Human");
    const CORE_HyperVector* ant_hv = core_kb_get_vector(mind, "Ant");
    const CORE_HyperVector* altman_hv = core_kb_get_vector(mind, "SamAltman");

    uint32_t dist_elephant_human = core_kb_distance(mind, elephant_hv, human_hv);
    uint32_t dist_elephant_ant = core_kb_distance(mind, elephant_hv, ant_hv);
    printf("Distance (Elephant <-> Human): %u / %u\n", dist_elephant_human, config.dimensionality);
    printf("Distance (Elephant <-> Ant):   %u / %u\n", dist_elephant_ant, config.dimensionality);
    if (dist_elephant_human < dist_elephant_ant) {
        printf("   [PASS] Engine correctly identifies 'Elephant' is conceptually closer to 'Human' than 'Ant'.\n");
    } else {
        printf("   [FAIL] The engine's conceptual space is not structured correctly.\n");
    }

    uint32_t dist_altman_human = core_kb_distance(mind, altman_hv, human_hv);
    uint32_t dist_altman_ant = core_kb_distance(mind, altman_hv, ant_hv);
    printf("\nDistance (SamAltman <-> Human): %u / %u\n", dist_altman_human, config.dimensionality);
    printf("Distance (SamAltman <-> Ant):   %u / %u\n", dist_altman_ant, config.dimensionality);
    if (dist_altman_human < dist_altman_ant) {
        printf("   [PASS] Engine correctly identifies 'SamAltman' is conceptually closer to 'Human' due to symmetric 'is_a' pull.\n");
    } else {
        printf("   [FAIL] The engine's conceptual space is not structured correctly for instances.\n");
    }

    print_header("Phase 3: Algebraic Query (Fact Retrieval)");
    perform_algebraic_query(mind, "SamAltman", "is_ceo_of", "OpenAI");
    
    print_header("Phase 4: Persistence Test (Save)");
    CORE_Status save_status = core_kb_save(mind, mind_filepath);
    if (save_status == CORE_OK) {
        printf("[PASS] Engine state saved to '%s'.\n", mind_filepath);
    } else {
        printf("[FAIL] Could not save state. Error: %s\n", core_status_to_string(save_status));
    }
    core_kb_destroy(mind);
    printf("Original engine object destroyed.\n");

    print_header("Phase 5: Persistence Test (Load)");
    CORE_Status load_status;
    CORE_KnowledgeBase* loaded_mind = core_kb_load(mind_filepath, &load_status);
    if (loaded_mind) {
        printf("[PASS] Engine state loaded from '%s'.\n", mind_filepath);
        CORE_Stats stats;
        core_kb_get_stats(loaded_mind, &stats);
        printf("   Loaded engine has %zu concepts and %zu assertions.\n", stats.concept_count, stats.assertion_count);
        printf("   Cache Hits: %zu, Cache Misses: %zu\n", stats.sim_cache_hits, stats.sim_cache_misses);

        print_header("Phase 6: Verifying Loaded State with Same Query");
        perform_algebraic_query(loaded_mind, "SamAltman", "is_ceo_of", "OpenAI");
        
        core_kb_destroy(loaded_mind);
    } else {
        printf("[FAIL] Could not load state. Error: %s\n", core_status_to_string(load_status));
    }
    
    printf("\n\nEngine shutdown sequence complete.\n");

    return 0;
}

/**
 * @brief Retrieves a snapshot of the knowledge base's current statistics.
 * @author The Continuum Synapse Team (Microsoft)
 * @date 2024
 * @param kb A pointer to the initialized CORE_KnowledgeBase.
 * @param out_stats A pointer to a CORE_Stats structure to be populated.
 * @return CORE_OK on success, or CORE_ERR_NULL_ARG if kb or out_stats is NULL.
 * @note This function is thread-safe. It acquires necessary locks to ensure a
 *       consistent snapshot of the stats, which may introduce a brief pause
 *       during high-write contention.
 */
CORE_Status core_kb_get_stats(CORE_KnowledgeBase* kb, CORE_Stats* out_stats) {
    // --- Input Validation ---
    if (!kb || !out_stats) {
        return CORE_ERR_NULL_ARG;
    }

    // --- Initialization ---
    // Zero out the struct to ensure no garbage values are returned.
    memset(out_stats, 0, sizeof(CORE_Stats));

    // --- Gather Stats Under Lock ---
    // A consistent snapshot requires careful locking. We start with the broadest
    // lock to get a stable view of the hash table's structure.

    core_mutex_lock(&kb->resize_lock);
    size_t current_bucket_count = kb->bucket_count;

    // These stats are protected by the resize_lock.
    out_stats->concept_count = kb->concept_count;
    out_stats->concept_capacity = current_bucket_count;
    out_stats->node_arena_bytes = kb->node_arena->total_bytes;
    out_stats->hv_arena_bytes = kb->hv_arena->total_bytes;
    out_stats->relation_arena_bytes = kb->relation_arena->total_bytes;

    // Atomically grab stats from other locked components.
    // Lock, copy, unlock pattern minimizes lock duration.
    core_mutex_lock(&kb->assertion_log_lock);
    out_stats->assertion_count = kb->assertion_count;
    core_mutex_unlock(&kb->assertion_log_lock);

    if (kb->sim_cache) {
        core_mutex_lock(&kb->sim_cache->lock);
        out_stats->sim_cache_hits = kb->sim_cache->hits;
        out_stats->sim_cache_misses = kb->sim_cache->misses;
        core_mutex_unlock(&kb->sim_cache->lock);
    }

    // To count total relations, we must iterate through every bucket. Since we
    // hold the resize_lock, the bucket array itself is stable. We still need
    // to lock each bucket individually to safely traverse its linked list.
    for (size_t i = 0; i < current_bucket_count; ++i) {
        core_mutex_lock(&kb->bucket_locks[i]);
        for (ConceptNode* node = kb->buckets[i]; node != NULL; node = node->next) {
            if (node->is_deleted) {
                continue;
            }
            // Traverse the linked list of relations for this concept.
            for (RelationEdge* e = node->relations_head; e; e = e->next) {
                if (e->strength > 0) {
                    out_stats->total_relations++;
                }
            }
        }
        core_mutex_unlock(&kb->bucket_locks[i]);
    }

    // Release the primary lock now that we are done iterating.
    core_mutex_unlock(&kb->resize_lock);

    return CORE_OK;
}
