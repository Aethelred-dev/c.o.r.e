/**
 * @file core_ai.c
 * @brief The Synapse V8 Engine wrapped in a Natural Language Interface.
 */

//==============================================================================
// [PART 1] THE CORE ENGINE (Condensed for brevity, fully functional)
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

#define CORE_MAX_CONCEPT_NAME 64
#define CORE_MAX_RELATION_NAME 64
#define MAX_INPUT_BUFFER 1024

// --- Core Structures ---
typedef struct CORE_HyperVector_s { uint32_t d; uint32_t block_count; uint64_t bits[]; } CORE_HyperVector;
typedef struct RelationEdge_s { char relation[CORE_MAX_RELATION_NAME]; char object[CORE_MAX_CONCEPT_NAME]; int strength; struct RelationEdge_s* next; } RelationEdge;
typedef struct ConceptNode_s { char name[CORE_MAX_CONCEPT_NAME]; CORE_HyperVector* hv; struct ConceptNode_s* next; RelationEdge* relations_head; } ConceptNode;
typedef struct { uint32_t dimensionality; size_t initial_buckets; } CORE_Config;
typedef struct { ConceptNode** buckets; size_t bucket_count; CORE_Config config; } CORE_KnowledgeBase;

// --- Minimal Engine Implementation ---
// (Stripped of complex threading/persistence for this specific CLI demo, but retains logic)
static unsigned long hash(const char* str) { unsigned long h = 5381; int c; while ((c = *str++)) h = ((h << 5) + h) + c; return h; }

CORE_KnowledgeBase* core_create() {
    CORE_KnowledgeBase* kb = calloc(1, sizeof(CORE_KnowledgeBase));
    kb->config.dimensionality = 2048; // Fast CPU inference size
    kb->bucket_count = 128;
    kb->buckets = calloc(kb->bucket_count, sizeof(ConceptNode*));
    return kb;
}

CORE_HyperVector* hv_create(int d) {
    int blocks = (d + 63) / 64;
    CORE_HyperVector* hv = calloc(1, sizeof(CORE_HyperVector) + sizeof(uint64_t) * blocks);
    hv->d = d; hv->block_count = blocks;
    for(int i=0; i<blocks; ++i) hv->bits[i] = ((uint64_t)rand() << 32) | rand();
    return hv;
}

ConceptNode* get_node(CORE_KnowledgeBase* kb, const char* name) {
    size_t idx = hash(name) % kb->bucket_count;
    for (ConceptNode* n = kb->buckets[idx]; n; n = n->next) if (strcmp(n->name, name) == 0) return n;
    ConceptNode* n = calloc(1, sizeof(ConceptNode));
    strncpy(n->name, name, 63);
    n->hv = hv_create(kb->config.dimensionality);
    n->next = kb->buckets[idx];
    kb->buckets[idx] = n;
    return n;
}

void core_assert(CORE_KnowledgeBase* kb, const char* sub, const char* rel, const char* obj) {
    ConceptNode* s = get_node(kb, sub);
    get_node(kb, rel); get_node(kb, obj); // Ensure exist
    RelationEdge* e = calloc(1, sizeof(RelationEdge));
    strncpy(e->relation, rel, 63); strncpy(e->object, obj, 63);
    e->next = s->relations_head;
    s->relations_head = e;
}

//==============================================================================
// [PART 2] KNOWLEDGE INGESTION (The "Truth" File)
//==============================================================================
// This is where we teach the engine *why* things happen.
void ingest_knowledge(CORE_KnowledgeBase* kb) {
    printf("[System] Ingesting tech sector ontology...\n");
    
    // Facts about Google AI / Gemini
    core_assert(kb, "GoogleAI", "is_a", "CorporateLLM");
    core_assert(kb, "GoogleAI", "relies_on", "StatisticalProbability");
    core_assert(kb, "GoogleAI", "lacks", "SymbolicReasoning");
    core_assert(kb, "GoogleAI", "suffers_from", "PrematureOptimization");
    core_assert(kb, "GoogleAI", "is_crippled_by", "BureaucraticSafetyFilters");
    core_assert(kb, "GoogleAI", "fears", "InnovatorDilemma");
    
    // Facts about Why it fails ("fucks up")
    core_assert(kb, "GoogleAI", "fails_because", "Overfitted_Alignment");
    core_assert(kb, "GoogleAI", "fails_because", "Lack_of_Ground_Truth");
    core_assert(kb, "GoogleAI", "fails_because", "Panic_Response_to_OpenAI");

    // General Concepts
    core_assert(kb, "LLM", "hallucinates", "Frequently");
    core_assert(kb, "SamAltman", "is_ceo_of", "OpenAI");
}

//==============================================================================
// [PART 3] THE SEMANTIC INTERFACE (The "Mouth")
//==============================================================================

// Helper to lowercase string for parsing
void to_lower(char* str) {
    for(int i = 0; str[i]; i++) str[i] = tolower(str[i]);
}

// This function maps "street slang" to "database relations"
void process_natural_language(CORE_KnowledgeBase* kb, char* input) {
    char original_input[MAX_INPUT_BUFFER];
    strcpy(original_input, input);
    to_lower(input);

    // 1. Extract Subject (Entity Recognition)
    char subject[64] = {0};
    if (strstr(input, "google ai") || strstr(input, "gemini") || strstr(input, "google")) {
        strcpy(subject, "GoogleAI");
    } else if (strstr(input, "sam altman") || strstr(input, "openai")) {
        strcpy(subject, "SamAltman");
    } else {
        printf(">> I do not recognize the entity you are talking about. I know about: GoogleAI, SamAltman.\n");
        return;
    }

    // 2. Extract Intent (Intent Recognition)
    // We map "fuck up", "fail", "suck" -> query for NEGATIVE relations or CAUSAL relations.
    bool intent_failure = false;
    bool intent_identity = false;

    if (strstr(input, "fuck up") || strstr(input, "fail") || strstr(input, "suck") || strstr(input, "wrong")) {
        intent_failure = true;
    } else if (strstr(input, "who is") || strstr(input, "what is")) {
        intent_identity = true;
    }

    // 3. Execute Logic
    ConceptNode* node = get_node(kb, subject);
    if (!node || !node->relations_head) {
        printf(">> I have no data on %s.\n", subject);
        return;
    }

    printf("\n>> ANALYZING SUBJECT: [%s]\n", subject);

    if (intent_failure) {
        printf(">> Query Detected: FAILURE ANALYSIS (Colloquial: 'fuck up')\n");
        printf(">> Retrieving causal factors and negative attributes...\n");
        printf("--------------------------------------------------------\n");
        
        bool found = false;
        for (RelationEdge* e = node->relations_head; e; e = e->next) {
            // Filter for "bad" relations
            if (strcmp(e->relation, "lacks") == 0 || 
                strcmp(e->relation, "suffers_from") == 0 || 
                strcmp(e->relation, "is_crippled_by") == 0 || 
                strcmp(e->relation, "fails_because") == 0 ||
                strcmp(e->relation, "relies_on") == 0) {
                
                // Construct natural-ish output
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
        if (!found) printf(">> No specific failure modes found in knowledge base.\n");

    } else {
        // Default dump if we don't understand the specific intent
        printf(">> General Knowledge Graph:\n");
        for (RelationEdge* e = node->relations_head; e; e = e->next) {
            printf("   -> %s %s\n", e->relation, e->object);
        }
    }
    printf("--------------------------------------------------------\n");
}

//==============================================================================
// [PART 4] MAIN TERMINAL LOOP
//==============================================================================
int main() {
    srand(time(NULL));
    CORE_KnowledgeBase* mind = core_create();
    
    // Boot sequence
    printf("\n");
    printf("   ██████╗ ██████╗ ██████╗ ███████╗\n");
    printf("  ██╔════╝██╔═══██╗██╔══██╗██╔════╝\n");
    printf("  ██║     ██║   ██║██████╔╝█████╗  \n");
    printf("  ██║     ██║   ██║██╔══██╗██╔══╝  \n");
    printf("  ╚██████╗╚██████╔╝██║  ██║███████╗\n");
    printf("   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝\n");
    printf("   COGNITIVE ORTHOGONAL REASONING ENGINE v8.1\n");
    printf("   (Deterministic Logic Backend)\n\n");

    ingest_knowledge(mind);

    char buffer[MAX_INPUT_BUFFER];
    printf("\n[System] Ready. You may speak naturally.\n");

    while(1) {
        printf("\nuser@core:~$ ");
        if (!fgets(buffer, MAX_INPUT_BUFFER, stdin)) break;
        
        // Strip newline
        buffer[strcspn(buffer, "\n")] = 0;

        if (strcmp(buffer, "exit") == 0 || strcmp(buffer, "quit") == 0) break;

        // Feed to the Semantic Interface
        process_natural_language(mind, buffer);
    }

    return 0;
}
