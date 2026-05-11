# Privacy and Safety Threat Model

## System Boundary

The demo is a local-first journaling prototype. The default analyzer runs in Python on the user's machine and the browser stores history in `localStorage`. No journal entry is sent to a third-party API by the shipped implementation.

## Protected Assets

- Raw journal text
- Emotion history and EHI trend
- Exported JSON, image, or PDF reports
- Any local transformer model files used through `SENTIMENT_MODEL_DIR`

## Main Risks and Mitigations

| Risk | Current mitigation | Future improvement |
| --- | --- | --- |
| Journal text leaves the device | Default analyzer is local and `/predict` runs on localhost | Package as desktop app or browser-only model |
| Browser history is readable by other local users | Clear-history button and no remote account | Optional passphrase encryption for local history |
| Over-trust in emotional feedback | UI and API include non-diagnostic disclaimer | Add first-run consent screen and clearer limitations |
| Self-harm language is under-detected | Keyword risk flag and urgent support message | Use validated crisis classifier and escalation resources |
| Exported reports leak sensitive text | Export is user-triggered only | Add redaction mode and export confirmation |
| Server logs expose entries | Built-in server suppresses request logging | Add explicit privacy tests and log audit |

## Safety Boundary

This project is not a clinical diagnosis, treatment, triage, or emergency response tool. It can support self-reflection, but it cannot determine whether a user is safe. When severe distress or self-harm language appears, the system should encourage immediate human support and local emergency resources.

## Data Retention

The default browser history remains only on the same device/browser profile until the user clears it. The backend does not persist entries.
