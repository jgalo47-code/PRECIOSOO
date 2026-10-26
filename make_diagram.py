import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
fig, ax = plt.subplots(figsize=(9, 6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
ax.add_patch(Rectangle((3, .3), 4.4, 6.4, fill=False)); ax.text(5.2, 6.45, "Campus Facility Reservation & Monitoring System", ha="center", weight="bold", fontsize=8)
uc = ["Upload & Validate Data", "View Reservation Statistics", "Analyze Cancellation Patterns", "Review Predicted Risk Levels", "Record Final Decision", "Manage Facility Reservations"]
pos = {}
for i, t in enumerate(uc):
    y = 5.6 - i*.95; ax.add_patch(Ellipse((5.2, y), 3.4, .75, fill=False)); ax.text(5.2, y, t, ha="center", va="center", fontsize=8); pos[t] = y
for a, x, y, ucs in [("Administrator", 9, 3.4, uc), ("User", 1, 1.6, [uc[5]])]:
    ax.plot(x, y+.3, "ko"); ax.plot([x, x], [y+.3, y-.4], "k"); ax.text(x, y-.8, a, ha="center", fontsize=9)
    for u in ucs: ax.plot([x, 3.5 if x < 5 else 6.9], [y, pos[u]], "k-", lw=.6)
plt.savefig("assets/use_case_diagram.png", dpi=150, bbox_inches="tight")
