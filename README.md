# Claim Relation Understanding and Extraction (CRUX)

This is a re-development of our system for the NIST TAC 2023 CRUX Track with a new design using a 2026 technology stack. To reproduce the system submitted to the TAC 2023 evaluation, please switch to the branch `nist-tac-2023-crux-workshop`.

## Background
In scenarios such as natural disasters or international conflicts, analysts and the public are often confronted with a vast amount of information coming through multiple media sources. In order to respond to these situations, there is a need for technologies that analyze and extract claims from multilingual multimedia sources and determine how the claims are related, to develop and maintain an understanding of events, situations, and trends around the world.

The Claim Relation Understanding and Extraction (CRUX) track evaluates the performance of systems on the tasks of claim frame extraction and cross-claim relation tagging from multilingual documents, using data developed under DARPA's Active Interpretation of Disparate Alternatives (AIDA) program. The goal of AIDA was to develop a multi-hypothesis semantic engine that generates explicit alternative interpretations of events, situations, and trends of interest from a variety of unstructured multimedia sources, for use in noisy, conflicting, and potentially deceptive information environments. In CRUX 2023, systems must extract claims about targeted topics in the COVID-19 Global Pandemic scenario. Participation in these tasks is open to all researchers who are interested in the tasks.

## Task Overview
**Definitions**: A claim is defined in relation to a claim template, which is a sentence that states, in general terms, a belief someone might hold about a topic within the COVID-19 Global Pandemic scenario. Within each claim template is an X Variable that can be filled by some piece of information. The following are examples of a claim template: X created SARS-CoV-2; X cures COVID-19. A claim, then, is a statement made implicitly or explicitly in or throughout a document that fills in the X Variable in a claim template. A claim also includes all of the informational context surrounding the claim, such as who is making the claim, when and where the claim was made, how the person making the claim feels about the claim, etc. At the center of a claim is the core claim, which refers to the root claim itself without any of the surrounding informational context. In the CRUX evaluation (as in AIDA), all relevant information about a single claim in a document is represented in a claim frame; in addition to what is being claimed, the claim frame includes other fields that correspond to the various pieces of information about the claim, including who is making the claim, when the claim is made, etc. For further details, please see the principles for claim frame annotation in the annotation data package documentation.

The CRUX track has two evaluation tasks:

- **Task 1**: Extract all claims about targeted topics from a corpus of documents: For each input document, find all claims about the targeted topics that are expressed in the document, and populate the claim frame for that claim with the information found in the document, following the annotation principles for claim frame annotation.
- **Task 2**: Identify cross-claim relations: For each given pair of claims, determine the relation between the two claims (identical, supporting, refuting, related, or unrelated).

**Documents**: Systems will process a set of 250 evaluation source documents in English, Spanish, and Russian. Each document may include both text and images.

**Evaluation**: System output will be scored by comparing against gold standard annotations for a subset of the documents.

## System Overview
