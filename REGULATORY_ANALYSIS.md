# Regulatory Analysis — Credit Scoring System

*Responsible AI & Data Ethics (PEL, SS2026). Week-1 deliverable: a regulatory
analysis for the project, mapped to the concrete system we built.*

This document maps our German-credit risk classifier to the regulatory
regime it would fall under if deployed in the EU, and shows where each legal
requirement is answered by a specific artifact in this repository. The point
is not abstract compliance theatre: every obligation below is tied to code,
a document, or a test.

---

## 1. Why this system is regulated at all

Credit scoring is one of the **named high-risk use cases** in the EU AI Act.
It is not a grey area. Annex III(5)(b) explicitly lists AI systems intended
to *"evaluate the creditworthiness of natural persons or establish their
credit score"* (excluding fraud detection) as high-risk. That single fact
drives most of the obligations below. In parallel, because the system makes
decisions about identifiable people using their personal and financial data,
the **GDPR** applies in full.

| Framework | Applies because… | Status for us |
|---|---|---|
| **EU AI Act** (Reg. 2024/1689) | Credit scoring = Annex III high-risk | Primary regime — analysed in §2 |
| **GDPR** (Reg. 2016/679) | Automated processing of personal data & profiling | Analysed in §3 |
| **EU Consumer Credit Directive** (2023/2225), Art. 18 | Creditworthiness assessment obligations | Context in §4 |
| **Equal-treatment / anti-discrimination law** (e.g. Racial Equality Dir. 2000/43, Gender Dir. 2004/113) | Protected-attribute disparities in a financial service | Motivates the fairness audit |
| **US Equal Credit Opportunity Act / "AI Bill of Rights"** | If deployed in the US | Comparative note in §5 |

---

## 2. EU AI Act — high-risk obligations (Chapter III)

The Act assigns duties along the value chain. As a *student project* we sit in
the **provider** role (we build the model); a bank deploying it would be the
**deployer**. Both sets of duties are listed because a credible pitch has to
show we understood the deployer's position too.

### Provider obligations (Arts. 8–17) mapped to our artifacts

| AI Act obligation | Article | How this project addresses it |
|---|---|---|
| **Risk management system** — continuous, iterative | Art. 9 | Risk register + treatments in [REPORT.md](REPORT.md) §5 and the notebook; adversarial threat model in [src/adversarial.py](src/adversarial.py) |
| **Data governance** — relevant, representative, checked for bias | Art. 10 | Data analysis + known-bias documentation in [REPORT.md](REPORT.md) §2 and [MODEL_CARD.md](MODEL_CARD.md) ("Known data risks"); missing-as-category decision documented in [src/preprocessing.py](src/preprocessing.py) |
| **Technical documentation** (Annex IV) | Art. 11 | This file + REPORT + MODEL_CARD together form the Annex IV pack |
| **Record-keeping / logging** — automatic event logging | Art. 12 | Design: each prediction is emitted with inputs, score and a SHAP explanation suitable for logging (see §7 of REPORT). *Gap:* a production logger is future work |
| **Transparency & information to deployers** | Art. 13 | [MODEL_CARD.md](MODEL_CARD.md): intended use, out-of-scope uses, performance, and fairness caveats |
| **Human oversight** — designed to be effectively overseen | Art. 14 | System is *decision support*: it outputs recommendation + explanation + fairness context; a loan officer holds final authority. Never issues an autonomous decision (MODEL_CARD "Intended Use") |
| **Accuracy, robustness & cybersecurity** | Art. 15 | Performance metrics (cost-sensitive recall) in REPORT §4; **robustness/security quantified** by the poisoning, evasion, and membership-inference studies in [src/adversarial.py](src/adversarial.py); reproducibility guarded by the test suite (`tests/`, 43 tests, 98.9% coverage) |

### Art. 10 in depth — the bias obligation

Art. 10(2)(f)–(g) requires examination for possible biases and their
mitigation. This is exactly what the fairness audit does:

- **Detection:** disparate impact, equal-opportunity gap and equalized-odds
  gap computed by Sex and Age band ([src/fairness_audit.py](src/fairness_audit.py)),
  and — on the full 20-feature UCI data — by **foreign-worker status**
  ([src/full_uci.py](src/full_uci.py)).
- **Finding:** the model **fails the 80% rule for age** (ratio 0.50) and for
  **foreign-worker status** (0.71), and is borderline for sex (0.80). The
  foreign-worker disparity is a direct anti-discrimination concern (a proxy for
  nationality/origin) that the 9-feature subset structurally cannot even surface
  — evidence that data governance and feature scope are themselves bias controls.
- **Mitigation:** group-specific thresholds repair the sex disparity
  (0.80 → 0.96) and the age disparity (0.50 → 0.82, halving the young-applicant
  wrongful-denial rate) — documented, owned decisions, exactly the "examination
  and mitigation of biases" the Article demands. For **foreign worker** the same
  technique lifts disparate impact (0.71 → 0.93) but collapses recall (0.76 →
  0.47) because that group is 96% of applicants; we therefore document it as a
  trade-off requiring a policy/reweighing decision rather than an automatic fix —
  itself an example of the Article's expectation that mitigation be *appropriate*,
  not merely applied.

### Art. 14 in depth — human oversight

The Act requires oversight to be *meaningful*, not a rubber-stamp. Our design
supports this by giving the human (a) the recommendation, (b) a per-decision
SHAP explanation of *why*, and (c) the group-level fairness context so the
officer knows the model under-serves young applicants and can weight its
output accordingly. This directly guards against automation bias.

### Prohibited-practice check (Art. 5)

We confirmed the system does **not** cross into prohibited territory: it is
not social scoring by a public authority, uses no subliminal or manipulative
techniques, and does not exploit vulnerabilities of a protected group. It is
high-risk, not prohibited.

---

## 3. GDPR — automated decisions and personal data

| GDPR provision | Requirement | Project response |
|---|---|---|
| **Art. 22** — automated individual decision-making | A person has the right not to be subject to a *solely* automated decision with legal/significant effect, and to obtain human intervention and an explanation | We deliberately keep a **human in the loop**, so the decision is not "solely automated". The SHAP local explanations ([shap_local_denied.png](results/shap_local_denied.png)) provide the "meaningful information about the logic" (Arts. 13–15) for an adverse-action notice |
| **Arts. 5 & 6** — lawfulness, purpose limitation, minimisation | Process only data needed for the stated purpose | Feature set is limited to credit-relevant attributes; sensitive attributes are *audited* rather than blindly used |
| **Art. 9** — special-category data | Extra protection for sensitive data | We use Sex and Age (not Art. 9 special categories as such), but treat them as **protected attributes** for fairness monitoring. We explicitly show that *dropping* them does **not** remove bias (proxies remain) — so "fairness through unawareness" is rejected on evidence |
| **Art. 25** — data protection by design & default | Bake in privacy | Missing account data is encoded as a category rather than harvesting more data; **membership-inference risk is measured** (attack AUC ≈ 0.53, mild) so privacy leakage is quantified, not assumed away |
| **Art. 35** — Data Protection Impact Assessment | High-risk processing needs a DPIA | This document + the model card + fairness audit constitute the substance a DPIA would require |

**Right to explanation, concretely.** GDPR's "logic involved" requirement is
often hand-waved. Here it is a real artifact: for any applicant the system
can produce a SHAP waterfall showing which features pushed the decision and
by how much — e.g. the denied applicant driven by low checking/saving
balances and a 24-month duration. That is a contestable, human-readable
reason, not a black box.

---

## 4. Consumer Credit Directive (2023/2225)

Art. 18 requires the creditworthiness assessment to be in the consumer's
interest, to avoid irresponsible lending, and — where it uses automated
processing — to give the consumer the right to human review and explanation.
This reinforces the same two controls we already built: **cost-sensitive
recall** (the model is tuned to catch genuinely risky applicants rather than
maximise approvals) and the **human-in-the-loop + explanation** design.

---

## 5. Comparative note — US

If deployed in the US, the **Equal Credit Opportunity Act (ECOA)** and its
Regulation B would require *specific, accurate* reasons for adverse action —
the CFPB has confirmed this applies even when a complex algorithm is used, so
"the model is too complicated to explain" is not a defence. Our SHAP local
explanations satisfy this. The White House **Blueprint for an AI Bill of
Rights** adds non-binding expectations (safe/effective systems, algorithmic
discrimination protections, notice and explanation, human alternatives) that
our fairness audit and human-oversight design already align with.

---

## 6. Compliance scorecard & residual gaps

| Area | State | Evidence |
|---|---|---|
| Data governance & bias examination | ✅ Done | fairness_audit.py, REPORT §2/§5 |
| Technical documentation (Annex IV) | ✅ Done | this file, REPORT, MODEL_CARD |
| Transparency / explainability | ✅ Done | explainability.py, SHAP artifacts |
| Human oversight design | ✅ Done | MODEL_CARD "Intended Use" |
| Accuracy, robustness & security | ✅ Done | adversarial.py, test suite (43 tests, 98.9% cov) |
| Fairness — sex | ✅ Mitigated | mitigation_comparison.csv (DI 0.80 → 0.96) |
| Fairness — age | ✅ Mitigated | mitigation_comparison.csv (DI 0.50 → 0.82); young-denial rate halved |
| Fairness — foreign worker | ⚠️ **Trade-off documented** | full_uci_mitigation.csv: threshold fix lifts DI 0.71 → 0.93 but recall collapses 0.76 → 0.47 (96% majority) → needs a policy/reweighing approach |
| Automatic event logging (Art. 12) | ⚠️ **Design-only** | logging hook not implemented |
| Post-market monitoring (Art. 72) | ⚠️ **Future** | needs a deployment to monitor |
| Conformity assessment / CE marking | ➖ N/A for a student project | would precede real deployment |

**Bottom line.** The system is engineered to be *compliant by construction*
for a high-risk credit-scoring use case: the legally hard parts — bias
examination, explainability, human oversight, robustness — are implemented
and evidenced — including implemented sex and age bias mitigation. The honest
residual gaps (a policy-level fix for foreign-worker fairness, and a production
logging/monitoring layer) are named explicitly rather than hidden, which is
itself what Art. 9's "continuous risk management" expects.
