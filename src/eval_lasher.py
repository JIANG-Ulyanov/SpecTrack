from rgbt import LasHeR

lasher = LasHeR()


lasher(
    tracker_name="SpecTrack",       
    result_path="/home/jiangchenxi/spectrack/RGBT_workspace/results/LasHeR/spectrack_b224",
    bbox_type="ltwh")

lasher(
    tracker_name="XTrack",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/XTrack_base/rgbt/lasher",
    bbox_type="ltwh")

lasher(
    tracker_name="SDSTrack",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/SDSTrack/lasher/cvpr2024_rgbt/SDSTrack_tracking_result",
    bbox_type="ltwh")

lasher(
    tracker_name="ViPT",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/VIPT/RGB-T/LasHeR/vipt_deep",
    bbox_type="ltwh")

lasher(
    tracker_name="Un-Track",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/UnTrack-Result/UnTrack-Uni-Thermal/LasHer",
    bbox_type="ltwh")


pr_dict = lasher.PR()
print("SpecTrack PR: \t", pr_dict["SpecTrack"][0])


npr_dict = lasher.NPR()
print("SpecTrack NPR: \t", npr_dict["SpecTrack"][0])


sr_dict = lasher.SR()
print("SpecTrack SR: \t", sr_dict["SpecTrack"][0])


lasher.draw_attributeRadar(metric_fun=lasher.PR, filename="lasher_PR_radar.png")
lasher.draw_attributeRadar(metric_fun=lasher.SR, filename="lasher_SR_radar.png")
