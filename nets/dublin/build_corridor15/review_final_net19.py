"""Inventory and map for the final 19-signal variant."""
import review_final_net18 as review
from build_net19 import NET,META,OUT

if __name__=="__main__":
    review.NET=NET;review.META=META;review.OUT=OUT
    review.main()
