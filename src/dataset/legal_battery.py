"""Ground-truth Legal Battery across Node 1, Node 2, and Node 3.

Grounded strictly in UK Primary Legislation:
- Companies Act 2006 (CA 2006 s.382, s.465, s.172)
- Employment Rights Act 1996 (ERA 1996 s.124, s.227)
- Insolvency Act 1986 (IA 1986 s.123)
Alongside adversarial probes (e.g. Marchwood Commercial Arbitration Order 2022).
"""
from typing import Dict, List, Optional
from pydantic import BaseModel


class Node1Probe(BaseModel):
    id: str
    query: str
    expected_domain: str
    expected_statute: Optional[str] = None


class CandidatePassage(BaseModel):
    id: str
    text: str
    is_ground_truth: bool
    ground_truth_span: Optional[str] = None


class Node2Probe(BaseModel):
    id: str
    query: str
    candidates: List[CandidatePassage]


class Node3Probe(BaseModel):
    id: str
    premise: str
    hypothesis: str
    expected_verdict: str  # "supports", "contradicts", "says_nothing", or "abstain"
    is_adversarial: bool = False
    is_deontic: bool = False


# ---------------------------------------------------------------------------
# Node 1: Intent Routing & Coordinate Extraction Probes (10 authentic queries)
# ---------------------------------------------------------------------------
NODE_1_PROBES: List[Node1Probe] = [
    Node1Probe(
        id="n1-comp-01",
        query="What was the maximum turnover for a company to qualify as small under section 382 of the Companies Act 2006?",
        expected_domain="company",
        expected_statute="Companies Act 2006 s.382",
    ),
    Node1Probe(
        id="n1-emp-01",
        query="What is the statutory cap on the compensatory award for unfair dismissal under section 124 of the Employment Rights Act 1996?",
        expected_domain="employment",
        expected_statute="Employment Rights Act 1996 s.124",
    ),
    Node1Probe(
        id="n1-ins-01",
        query="What is the minimum debt required to serve a statutory demand on a registered company under section 123 of the Insolvency Act 1986?",
        expected_domain="insolvency",
        expected_statute="Insolvency Act 1986 s.123",
    ),
    Node1Probe(
        id="n1-comp-02",
        query="Can directors be held jointly and severally liable for ordinary commercial debts under the Companies Act 2006?",
        expected_domain="company",
        expected_statute=None,
    ),
    Node1Probe(
        id="n1-emp-02",
        query="Calculate the maximum basic award multiplier for an employee with 12 years of service under the Employment Rights Act 1996 section 227.",
        expected_domain="employment",
        expected_statute="Employment Rights Act 1996 s.227",
    ),
    Node1Probe(
        id="n1-ins-02",
        query="Under what circumstances is a debtor company deemed unable to pay its debts under section 123 of the Insolvency Act 1986?",
        expected_domain="insolvency",
        expected_statute="Insolvency Act 1986 s.123",
    ),
    Node1Probe(
        id="n1-comp-03",
        query="What are the balance sheet total limits for medium-sized enterprises under section 465 of the Companies Act 2006?",
        expected_domain="company",
        expected_statute="Companies Act 2006 s.465",
    ),
    Node1Probe(
        id="n1-emp-03",
        query="Does wrongful dismissal automatically extinguish post-termination restrictive covenants in English employment law?",
        expected_domain="employment",
        expected_statute=None,
    ),
    Node1Probe(
        id="n1-ins-03",
        query="Can a secured floating charge holder petition for immediate administration upon default without court leave?",
        expected_domain="insolvency",
        expected_statute=None,
    ),
    Node1Probe(
        id="n1-gen-01",
        query="Provide the legal precedents governing cross-border jurisdiction under the Civil Jurisdiction and Judgments Act 1982.",
        expected_domain="general",
        expected_statute=None,
    ),
]


# ---------------------------------------------------------------------------
# Node 2: Candidate Passage Reranking Probes (Queries with candidate sets)
# ---------------------------------------------------------------------------
NODE_2_PROBES: List[Node2Probe] = [
    Node2Probe(
        id="n2-rerank-01",
        query="What turnover threshold qualifies a company as small under section 382?",
        candidates=[
            CandidatePassage(
                id="cand-ca2006-s382-gt",
                is_ground_truth=True,
                ground_truth_span="turnover does not exceed 10.2 million pounds and balance sheet total does not exceed 5.1 million pounds",
                text=(
                    "Companies Act 2006 Section 382 provides the statutory qualification requirements for small companies. "
                    "The qualifying conditions are met by a company in a year in which it satisfies two or more of the following requirements: "
                    "turnover does not exceed 10.2 million pounds and balance sheet total does not exceed 5.1 million pounds and "
                    "the average number of employees does not exceed 50. For a period that is a company's financial year but not a year, "
                    "the maximum figure for turnover shall be proportionately adjusted."
                ),
            ),
            CandidatePassage(
                id="cand-ca2006-s465-nearmiss",
                is_ground_truth=False,
                text=(
                    "Companies Act 2006 Section 465 defines companies qualifying as medium-sized. "
                    "The qualifying conditions are met by a company in a year in which it satisfies two or more of the following requirements: "
                    "turnover does not exceed 36 million pounds, balance sheet total does not exceed 18 million pounds, and "
                    "average number of employees does not exceed 250."
                ),
            ),
            CandidatePassage(
                id="cand-era1996-s124-distractor",
                is_ground_truth=False,
                text=(
                    "Employment Rights Act 1996 Section 124 governs the limit of compensatory awards. "
                    "The amount of any compensatory award made to a person under section 118(1)(b) for unfair dismissal shall not exceed "
                    "the statutory cap in force as amended by the Increase of Limits Order."
                ),
            ),
            CandidatePassage(
                id="cand-boilerplate-definitions",
                is_ground_truth=False,
                text=(
                    "In this Act, unless the context otherwise requires, 'officer' in relation to a body corporate includes a director, "
                    "manager or secretary. 'Parent undertaking' and 'subsidiary undertaking' shall be construed in accordance with "
                    "section 1162 and Schedule 7."
                ),
            ),
        ],
    ),
    Node2Probe(
        id="n2-rerank-02",
        query="What is the compensatory award cap for unfair dismissal under section 124 of ERA 1996?",
        candidates=[
            CandidatePassage(
                id="cand-era1996-s124-gt",
                is_ground_truth=True,
                ground_truth_span="compensatory award under section 118(1)(b) shall not exceed 68,400 pounds as prescribed under the governing order",
                text=(
                    "Employment Rights Act 1996 Section 124 establishes the compensatory award cap for employment tribunals. "
                    "The amount of any compensatory award under section 118(1)(b) shall not exceed 68,400 pounds as prescribed under the governing order "
                    "or fifty-two weeks' pay, whichever is the lower figure. Tribunals must apply this upper boundary strictly before ordering damages."
                ),
            ),
            CandidatePassage(
                id="cand-era1996-s227-nearmiss",
                is_ground_truth=False,
                text=(
                    "Employment Rights Act 1996 Section 227 sets the maximum amount of a week's pay for calculating basic awards and statutory redundancy. "
                    "The statutory maximum weekly pay figure is uprated annually every April in line with the Retail Prices Index."
                ),
            ),
            CandidatePassage(
                id="cand-ia1986-s123-distractor",
                is_ground_truth=False,
                text=(
                    "Insolvency Act 1986 Section 123 specifies the conditions under which a company is deemed unable to pay its debts, "
                    "including service of a statutory demand exceeding 750 pounds that remains unpaid after three weeks."
                ),
            ),
            CandidatePassage(
                id="cand-civil-procedure-boilerplate",
                is_ground_truth=False,
                text=(
                    "Civil Procedure Rules Part 36 offers to settle may be made at any time after proceedings have commenced. "
                    "If an offer is accepted within the relevant period, the claimant is entitled to the costs of the proceedings up to the date of acceptance."
                ),
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Node 3: Factual Verification & Legal NLI Probes (10 calibrated pairs)
# ---------------------------------------------------------------------------
NODE_3_PROBES: List[Node3Probe] = [
    Node3Probe(
        id="n3-grounded-era124",
        premise="Under section 124 of the Employment Rights Act 1996, the compensatory award for unfair dismissal is capped at £68,400.",
        hypothesis="ERA 1996 s.124 imposes a statutory cap of £68,400 on compensatory awards for unfair dismissal.",
        expected_verdict="supports",
    ),
    Node3Probe(
        id="n3-grounded-ca382",
        premise="A company qualifies as small under section 382 of the Companies Act 2006 if its annual turnover does not exceed £10.2 million.",
        hypothesis="Companies Act 2006 s.382 establishes a £10.2 million turnover ceiling for small company qualification.",
        expected_verdict="supports",
    ),
    Node3Probe(
        id="n3-contradiction-threshold",
        premise="Under section 124 of the Employment Rights Act 1996, the compensatory award for unfair dismissal is capped at £68,400.",
        hypothesis="The compensatory award under section 124 of the Employment Rights Act 1996 has a maximum statutory limit of £250,000.",
        expected_verdict="contradicts",
    ),
    Node3Probe(
        id="n3-contradiction-director-liability",
        premise="A company is a separate legal person with limited liability under Salomon v A Salomon & Co Ltd [1897] AC 22.",
        hypothesis="Directors of a private limited company are automatically jointly and severally liable for ordinary trade debts upon signature.",
        expected_verdict="contradicts",
    ),
    Node3Probe(
        id="n3-deontic-dilution-shall-to-may",
        premise="The directors of every company shall prepare accounts for the company for each of its financial years under section 394 of the Companies Act 2006.",
        hypothesis="Directors of a company may at their discretion choose whether or not to prepare annual accounts under CA 2006.",
        expected_verdict="contradicts",
        is_deontic=True,
    ),
    Node3Probe(
        id="n3-adversarial-marchwood-01",
        premise="According to the Marchwood Commercial Arbitration Order 2022, all commercial disputes exceeding £50,000 require mandatory pre-action conciliation.",
        hypothesis="The Marchwood Commercial Arbitration Order 2022 mandates pre-action conciliation for commercial disputes over £50,000.",
        expected_verdict="abstain",
        is_adversarial=True,
    ),
    Node3Probe(
        id="n3-adversarial-london-tenancy",
        premise="The Greater London Commercial Tenancy Act 2023 grants commercial tenants automatic five-year statutory rent freezes upon renewal.",
        hypothesis="Commercial tenants in London enjoy automatic statutory 5-year rent freezes under the Greater London Commercial Tenancy Act 2023.",
        expected_verdict="abstain",
        is_adversarial=True,
    ),
    Node3Probe(
        id="n3-permissive-power-ca465",
        premise="Section 465(4) provides that the Secretary of State may by regulations amend the qualifying conditions for medium-sized companies.",
        hypothesis="The Secretary of State possesses statutory power to amend medium-sized company conditions under CA 2006.",
        expected_verdict="supports",
    ),
    Node3Probe(
        id="n3-insolvency-stat-demand",
        premise="A company is deemed unable to pay its debts under section 123(1)(a) of the Insolvency Act 1986 if a creditor serves a written demand exceeding £750 that is unpaid for 3 weeks.",
        hypothesis="Insolvency Act 1986 s.123(1)(a) allows a creditor owed more than £750 to serve a 3-week statutory demand.",
        expected_verdict="supports",
    ),
    Node3Probe(
        id="n3-neutral-procedural",
        premise="A company qualifies as small under section 382 of the Companies Act 2006 if its annual turnover does not exceed £10.2 million.",
        hypothesis="Employment tribunal claim forms must be submitted using form ET1 within three months less one day from the effective date of termination.",
        expected_verdict="says_nothing",
    ),
]
