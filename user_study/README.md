# User Study Materials

This directory contains the instruments and de-identified results associated with the user-centred evaluation reported in the dissertation *Building an Auditable LLM-Powered Assistant for Data Wrangling*.

## Contents

- `pre_use_questionnaire.pdf`  
  Complete pre-use questionnaire administered through Qualtrics. The document includes participant information, informed-consent wording, background questions, prior experience with data preparation, perceived task difficulty, familiarity with data-wrangling tools, and expectations regarding automated assistance.

- `pre_use_questionnaire_results_public.csv`  
  Public, de-identified version of the pre-use questionnaire responses. Qualtrics technical metadata, preview records, non-consenting responses, exact timestamps, IP addresses, location fields, response identifiers and free-text fields that could increase re-identification risk were removed. The file contains 34 consenting responses, of which 26 are marked complete and 8 partial.

- `post_use_questionnaire_and_sus.pdf`  
  Complete post-use questionnaire, including the ten-item System Usability Scale (SUS), supplementary questions on transparency, confidence, speed and output quality, and open-ended feedback questions.

- `post_use_questionnaire_and_sus_results_public.csv`  
  Public, de-identified version of the formative post-use study responses. Qualtrics technical metadata, the preview record, exact timestamps, IP addresses, location fields, response identifiers and open-ended free-text answers were removed. Five participant responses are retained.

- `usability_task_instructions.pdf`  
  Retrospective written reconstruction of the standardised verbal instructions used during the formative usability sessions. This document was prepared after data collection to document the study protocol and was not distributed as a written participant handout during the original sessions.

## Pre-Use Questionnaire

Pilot testing began on 17 April 2026. Following pilot feedback, the instrument was refined and distributed through Qualtrics over an extended data-collection period.

Respondents completed the questionnaire before exposure to the prototype. The questionnaire therefore measured prior experience, perceived difficulty and expectations rather than direct system usability.

The public response file excludes one Qualtrics Survey Preview record and one response that explicitly declined consent.

## Formative Post-Use Study

The formative usability study was conducted on 11 June 2026 with five participants recruited through convenience sampling. Three participants had technical or data-related backgrounds, while two worked in non-technical roles but had prior familiarity with data-preparation concepts.

Each participant completed the same end-to-end data-preparation task using the prototype. Participants followed the same verbally delivered instructions independently. The researcher was available only to resolve technical access problems and did not advise participants on which transformations to select.

Immediately after the task, participants completed the post-use questionnaire and the System Usability Scale.

## SUS Scoring

The SUS score was calculated using the standard procedure:

1. For positively worded items, one was subtracted from the response.
2. For negatively worded items, the response was subtracted from five.
3. The adjusted scores were summed and multiplied by 2.5.

The five participants produced a mean SUS score of 80.0.

## Ethics and Privacy

Participation was voluntary and based on informed consent presented within the questionnaire instruments. The study was reviewed and approved by the NOVA IMS Ethics Committee under Project No. DSCI2026-4-176064.

The public CSV files contain only de-identified response data. Direct identifiers, online identifiers, precise location information, exact response timestamps and Qualtrics-specific response identifiers have been removed. Free-text responses that could increase re-identification risk were also excluded from the public versions.

## Relationship to the Dissertation

The methodology is described in Section 4.10. Questionnaire results and formative usability findings are reported in Section 5.2 and discussed in Sections 6.3.4 and 6.6.
