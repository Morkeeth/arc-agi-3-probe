import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Source: runs/duck-control/summary.txt and runs/duck-deadprobe/summary.txt
games=["tu93","sc25","cd82","ls20","g50t"]
ctrl=[5.40,0.94,0.00,0.00,0.00]; ctrl_lv=[2,2,0,0,0]
grd=[0.51,0.00,0.71,0.55,0.00]; grd_lv=[2,0,1,1,0]
skips=[84,19,35,0,8]
BG="#fcfcfb"; T1="#0b0b0b"; T2="#52514e"; C1="#2a78d6"; C2="#eb6834"
fig,ax=plt.subplots(figsize=(8,4.5),dpi=200); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
x=range(len(games)); w=0.36
b1=ax.bar([i-w/2-0.01 for i in x],ctrl,w,color=C1,label="Plain Duck (submitted)")
b2=ax.bar([i+w/2+0.01 for i in x],grd,w,color=C2,label="Duck + repeat-no-op guard")
for bars,lv in ((b1,ctrl_lv),(b2,grd_lv)):
    for b,l in zip(bars,lv):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.08,f"{b.get_height():.2f}\n{l} lvl",ha="center",va="bottom",fontsize=7.5,color=T2)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{g}\n{s} guard skips" for g,s in zip(games,skips)],fontsize=8.5,color=T1)
ax.set_ylabel("Game score (Kaggle log)",color=T2,fontsize=9); ax.set_ylim(0,7)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color("#d6d5d0")
ax.tick_params(colors=T2,labelsize=8); ax.yaxis.grid(True,color="#ecebe7",lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=8.5,loc="upper right")
ax.set_title("One run per arm, 5 public ARC-AGI-3 games. Mean 1.27 vs 0.35, 4 levels each.",fontsize=10,color=T1,loc="left")
fig.tight_layout(); fig.savefig("ARC-PAPER-FIG1-AB.png",facecolor=BG)
