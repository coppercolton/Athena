"""Print Athena's conservative AGI readiness audit."""

from athena.readiness import EvidenceLevel, assess_agi_readiness


report = assess_agi_readiness()
print("Athena AGI readiness audit")
print(f"AGI ready: {report.agi_ready}")
print("No composite percentage is reported; every broad gate must pass.\n")
for gate in report.gates:
    marker = (
        "PASS"
        if gate.evidence is EvidenceLevel.DEMONSTRATED
        else "LAB "
        if gate.evidence is EvidenceLevel.NARROW
        else "MISS"
    )
    print(f"[{marker}] {gate.question}")
    print(f"       {gate.current_evidence}")

counts = report.counts()
print("\nEvidence summary")
for label, count in counts.items():
    print(f"  {label}: {count}")
