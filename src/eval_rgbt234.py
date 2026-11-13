from rgbt import RGBT234
import time


st = time.time()
rgbt234 = RGBT234()


rgbt234(
    tracker_name="SpecTrack",      
    result_path="/home/jiangchenxi/spectrack/RGBT_workspace/results/RGBT234/spectrack_b224",

    bbox_type="ltwh")

rgbt234(
    tracker_name="XTrack",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/XTrack_base/rgbt/rgbt234",
    bbox_type="ltwh")

rgbt234(
    tracker_name="SDSTrack",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/SDSTrack/rgbt234/cvpr2024_rgbt/SDSTrack",
    bbox_type="ltwh")

rgbt234(
    tracker_name="ViPT",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/VIPT/RGB-T/RGBT234/vipt_deep",
    bbox_type="ltwh")

rgbt234(
    tracker_name="Un-Track",
    result_path="/home/jiangchenxi/spectrack/plot/tracking_result/UnTrack-Result/UnTrack-Uni-Thermal/RGBT234",
    bbox_type="ltwh")


pr_dict,_ = rgbt234.MPR("SpecTrack")
print("SpecTrack MPR: \t", pr_dict)


apf_sr,_ = rgbt234.MSR("SpecTrack")
print("SpecTrack MSR: \t", apf_sr)


print(f"Time: {time.time()-st}")    


pr_tc_dict = rgbt234.MPR(seqs=rgbt234.TC)
sr_tc_dict = rgbt234.MSR(seqs=rgbt234.TC)


rgbt234.draw_attributeRadar(metric_fun=rgbt234.MPR, filename="RGBT234_MPR_radar.png")
rgbt234.draw_attributeRadar(metric_fun=rgbt234.MSR, filename="RGBT234_MSR_radar.png")     


print(f"Time: {time.time()-st}")    
