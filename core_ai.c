/**
 * @file core_ai.c
 * @author The Aethelred Team
 *
 * @copyright Copyright (c) 2025 The Aethelred Team.
 *
 * This file implements a mock Hyperdimensional Computing (HDC) engine 
 * capable of ingesting semantic triples and responding to natural language 
 * queries.
 * 
 * @details
 * Implementation uses a hash-map based graph storage.
 * Vector operations are simulated for CPU efficiency in this build.
 */

//==============================================================================
// INCLUDES & DEFINITIONS
//==============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include <ctype.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Configuration Constants
#define CORE_MAX_CONCEPT_NAME    64
#define CORE_MAX_RELATION_NAME   64
#define MAX_INPUT_BUFFER         1024

//==============================================================================
// [PART 1] CORE DATA STRUCTURES
//==============================================================================

/**
 * @brief Represents a Semantic Hypervector.
 * Using a flexible array member for bit storage to allow dynamic dimensionality.
 */
typedef struct CORE_HyperVector_s 
{ 
    uint32_t    d;              // Dimensionality (e.g., 2048)
    uint32_t    block_count;    // Number of uint64 blocks required
    uint64_t    bits[];         // Flexible array for bitwise operations
} CORE_HyperVector;

/**
 * @brief Represents a directed edge in the Knowledge Graph.
 * Stores the predicate (relation) and the target (object).
 */
typedef struct RelationEdge_s 
{ 
    char                    relation[CORE_MAX_RELATION_NAME]; 
    char                    object[CORE_MAX_CONCEPT_NAME]; 
    int                     strength; 
    struct RelationEdge_s*  next; 
} RelationEdge;

/**
 * @brief A node representing a distinct Concept or Entity.
 * Contains the semantic ID (Hypervector) and a list of outgoing relations.
 */
typedef struct ConceptNode_s 
{ 
    char                    name[CORE_MAX_CONCEPT_NAME]; 
    CORE_HyperVector*       hv; 
    struct ConceptNode_s*   next;           // Linked list for hash collisions
    RelationEdge*           relations_head; // Adjacency list
} ConceptNode;

/**
 * @brief Engine Configuration.
 */
typedef struct 
{ 
    uint32_t    dimensionality; 
    size_t      initial_buckets; 
} CORE_Config;

/**
 * @brief Main Knowledge Base Handle.
 * Contains the hash table buckets and configuration.
 */
typedef struct 
{ 
    ConceptNode**   buckets; 
    size_t          bucket_count; 
    CORE_Config     config; 
} CORE_KnowledgeBase;

//==============================================================================
// CORE ENGINE IMPLEMENTATION
//==============================================================================

/**
 * @brief DJB2 Hash Function.
 * Standard string hashing algorithm for bucket distribution.
 */
static unsigned long hash(const char* str) 
{ 
    unsigned long h = 5381; 
    int c; 
    while ((c = *str++)) 
    {
        h = ((h << 5) + h) + c; 
    }
    return h; 
}

/**
 * @brief Initializes the core knowledge base engine.
 * Allocates memory for the main structure and hash buckets.
 */
CORE_KnowledgeBase* core_create() 
{
    CORE_KnowledgeBase* kb = calloc(1, sizeof(CORE_KnowledgeBase));
    
    // Configuration defaults
    kb->config.dimensionality = 2048; // Tuned for fast CPU inference
    kb->bucket_count = 128;
    
    kb->buckets = calloc(kb->bucket_count, sizeof(ConceptNode*));
    return kb;
}

/**
 * @brief Creates a random Hypervector.
 * In a real HDC system, these would be orthogonal; here we use random noise.
 */
CORE_HyperVector* hv_create(int d) 
{
    int blocks = (d + 63) / 64;
    CORE_HyperVector* hv = calloc(1, sizeof(CORE_HyperVector) + sizeof(uint64_t) * blocks);
    
    hv->d = d; 
    hv->block_count = blocks;
    
    for(int i = 0; i < blocks; ++i) 
    {
        // Generate 64-bit random noise
        hv->bits[i] = ((uint64_t)rand() << 32) | rand();
    }
    
    return hv;
}

/**
 * @brief Retrieves a Concept Node by name, or creates it if it doesn't exist.
 * Implements Get-Or-Add logic with separate chaining for collisions.
 */
ConceptNode* get_node(CORE_KnowledgeBase* kb, const char* name) 
{
    size_t idx = hash(name) % kb->bucket_count;
    
    // Traverse bucket linked list to find existing node
    for (ConceptNode* n = kb->buckets[idx]; n; n = n->next) 
    {
        if (strcmp(n->name, name) == 0) 
        {
            return n;
        }
    }
    
    // Node not found: Create new
    ConceptNode* n = calloc(1, sizeof(ConceptNode));
    strncpy(n->name, name, 63);
    n->hv = hv_create(kb->config.dimensionality);
    
    // Insert at head of bucket
    n->next = kb->buckets[idx];
    kb->buckets[idx] = n;
    
    return n;
}

/**
 * @brief Asserts a fact (triple) into the knowledge base.
 * Format: Subject -> Relation -> Object
 */
void core_assert(CORE_KnowledgeBase* kb, const char* sub, const char* rel, const char* obj) 
{
    ConceptNode* s = get_node(kb, sub);
    
    // Ensure relation and object nodes exist in the graph (lazy instantiation)
    get_node(kb, rel); 
    get_node(kb, obj); 
    
    RelationEdge* e = calloc(1, sizeof(RelationEdge));
    strncpy(e->relation, rel, 63); 
    strncpy(e->object, obj, 63);
    
    // Prepend to relation list (O(1) insertion)
    e->next = s->relations_head;
    s->relations_head = e;
}

//==============================================================================
// [PART 2] KNOWLEDGE INGESTION
//==============================================================================

/**
 * @brief Populates the knowledge base with the "Ground Truth".
 * Defines the ontology for the tech sector simulation.
 */
void ingest_knowledge(CORE_KnowledgeBase* kb) 
{
    printf("[System] Ingesting tech sector ontology...\n");
    
    // --- Google AI / Gemini Ontology ---
    core_assert(kb, "GoogleAI", "is_a",             "CorporateLLM");
    core_assert(kb, "GoogleAI", "relies_on",        "StatisticalProbability");
    core_assert(kb, "GoogleAI", "lacks",            "SymbolicReasoning");
    core_assert(kb, "GoogleAI", "suffers_from",     "PrematureOptimization");
    core_assert(kb, "GoogleAI", "is_crippled_by",   "BureaucraticSafetyFilters");
    core_assert(kb, "GoogleAI", "fears",            "InnovatorDilemma");
    
    // --- Root Cause Analysis (Failure Modes) ---
    core_assert(kb, "GoogleAI", "fails_because",    "Overfitted_Alignment");
    core_assert(kb, "GoogleAI", "fails_because",    "Lack_of_Ground_Truth");
    core_assert(kb, "GoogleAI", "fails_because",    "Panic_Response_to_OpenAI");

    // --- General Market Concepts ---
    core_assert(kb, "LLM",          "hallucinates", "Frequently");
    core_assert(kb, "SamAltman",    "is_ceo_of",    "OpenAI");
}

//==============================================================================
// [PART 3] SEMANTIC INTERFACE (NLP LAYER)
//==============================================================================

/**
 * @brief Utility: Converts string to lowercase in place.
 */
void to_lower(char* str) 
{
    for(int i = 0; str[i]; i++) 
    {
        str[i] = tolower(str[i]);
    }
}

/**
 * @brief NLP Processor.
 * Maps colloquial user input ("street slang") to database queries.
 * 
 * @param kb    Pointer to Knowledge Base
 * @param input User input string
 */
void process_natural_language(CORE_KnowledgeBase* kb, char* input) 
{
    char original_input[MAX_INPUT_BUFFER];
    strcpy(original_input, input);
    to_lower(input);

    // ---------------------------------------------------------
    // 1. Entity Recognition (Keyword matching)
    // ---------------------------------------------------------
    char subject[64] = {0};
    
    if (strstr(input, "google ai") || strstr(input, "gemini") || strstr(input, "google")) 
    {
        strcpy(subject, "GoogleAI");
    } 
    else if (strstr(input, "sam altman") || strstr(input, "openai")) 
    {
        strcpy(subject, "SamAltman");
    } 
    else 
    {
        printf(">> I do not recognize the entity you are talking about. I know about: GoogleAI, SamAltman.\n");
        return;
    }

    // ---------------------------------------------------------
    // 2. Intent Recognition
    // ---------------------------------------------------------
    bool intent_failure = false;
    bool intent_identity = false;

    // Detect colloquialisms for failure/error states
    if (strstr(input, "fuck up") || strstr(input, "fail") || strstr(input, "suck") || strstr(input, "wrong")) 
    {
        intent_failure = true;
    } 
    else if (strstr(input, "who is") || strstr(input, "what is")) 
    {
        intent_identity = true;
    }

    // ---------------------------------------------------------
    // 3. Logic Execution & Graph Traversal
    // ---------------------------------------------------------
    ConceptNode* node = get_node(kb, subject);
    if (!node || !node->relations_head) 
    {
        printf(">> I have no data on %s.\n", subject);
        return;
    }

    printf("\n>> ANALYZING SUBJECT: [%s]\n", subject);

    if (intent_failure) 
    {
        printf(">> Query Detected: FAILURE ANALYSIS (Colloquial: 'fuck up')\n");
        printf(">> Retrieving causal factors and negative attributes...\n");
        printf("--------------------------------------------------------\n");
        
        bool found = false;
        for (RelationEdge* e = node->relations_head; e; e = e->next) 
        {
            // Filter for relations that indicate negative sentiment or causality
            if (strcmp(e->relation, "lacks") == 0 || 
                strcmp(e->relation, "suffers_from") == 0 || 
                strcmp(e->relation, "is_crippled_by") == 0 || 
                strcmp(e->relation, "fails_because") == 0 ||
                strcmp(e->relation, "relies_on") == 0) 
            {
                // Format output based on specific relation type
                if (strcmp(e->relation, "fails_because") == 0)
                    printf("   -> REASON: %s\n", e->object);
                else if (strcmp(e->relation, "lacks") == 0)
                    printf("   -> MISSING COMPONENT: %s\n", e->object);
                else if (strcmp(e->relation, "is_crippled_by") == 0)
                    printf("   -> SYSTEMIC BLOCKER: %s\n", e->object);
                else
                    printf("   -> %s: %s\n", e->relation, e->object);
                
                found = true;
            }
        }
        
        if (!found) 
        {
            printf(">> No specific failure modes found in knowledge base.\n");
        }

    } 
    else 
    {
        // Default: Dump all knowledge if intent is generic
        printf(">> General Knowledge Graph:\n");
        for (RelationEdge* e = node->relations_head; e; e = e->next) 
        {
            printf("   -> %s %s\n", e->relation, e->object);
        }
    }
    printf("--------------------------------------------------------\n");
}

//==============================================================================
// [PART 4] MAIN ENTRY POINT
//==============================================================================
int main() 
{
    srand(time(NULL));
    
    // Initialize Engine
    CORE_KnowledgeBase* mind = core_create();
    
    // Render Boot Splash
    printf("\n");
    printf("   ██████╗ ██████╗ ██████╗ ███████╗\n");
    printf("  ██╔════╝██╔═══██╗██╔══██╗██╔════╝\n");
    printf("  ██║     ██║   ██║██████╔╝█████╗  \n");
    printf("  ██║     ██║   ██║██╔══██╗██╔══╝  \n");
    printf("  ╚██████╗╚██████╔╝██║  ██║███████╗\n");
    printf("   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝\n");
    printf("   COGNITIVE ORTHOGONAL REASONING ENGINE v8.1\n");
    printf("   (Deterministic Logic Backend)\n\n");

    // Load Data
    ingest_knowledge(mind);

    char buffer[MAX_INPUT_BUFFER];
    printf("\n[System] Ready. You may speak naturally.\n");

    // Main Event Loop
    while(1) 
    {
        printf("\nuser@core:~$ ");
        if (!fgets(buffer, MAX_INPUT_BUFFER, stdin)) 
        {
            break;
        }
        
        // Strip newline character
        buffer[strcspn(buffer, "\n")] = 0;

        if (strcmp(buffer, "exit") == 0 || strcmp(buffer, "quit") == 0) 
        {
            break;
        }

        // Feed to the Semantic Interface
        process_natural_language(mind, buffer);
    }

    return 0;
}
