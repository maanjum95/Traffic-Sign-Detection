import subprocess
import seeed_python_reterminal.core as rt
import seeed_python_reterminal.button as rt_btn

print("Startup script for traffic sign detector.")

device = rt.get_button_device()
algo_popen = None

for event in device.read_loop():
    buttonEvent = rt_btn.ButtonEvent(event)
    if buttonEvent.name and buttonEvent.name == rt_btn.ButtonName.O:
        if buttonEvent.value:
            if algo_popen is None:
                print("Starting the demo...")
                algo_popen = subprocess.Popen(["python3", "/home/pi/traffic_sign_detection/demo.py"])
                print("Demo started")
            else:
                print("Stopping the algorithm...")
                algo_popen.kill()
                algo_popen = None
                print("Stopped the algorithm.")

    
