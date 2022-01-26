import asyncio
import threading
import seeed_python_reterminal.core as rt
import seeed_python_reterminal.button as rt_btn


def setup_btns(model_params):
    btn_device = rt.get_button_device()

    thread = threading.Thread(target=btn_coroutine, args=(btn_device, model_params))
    thread.start()

    #loop.run_forever()

# F1: shows the bbox
# F2: shows the labels
# F3: shows the signs
def btn_coroutine(device, model_params):
    for event in device.read_loop():
        buttonEvent = rt_btn.ButtonEvent(event)
        if buttonEvent.name != None and buttonEvent.value:
            print(f"name={str(buttonEvent.name)} value={buttonEvent.value}")
            if buttonEvent.name == rt_btn.ButtonName.F1:
                model_params["show_bbox"] = not model_params["show_bbox"]
            elif buttonEvent.name == rt_btn.ButtonName.F2:
                model_params["show_label"] = not model_params["show_label"]
            elif buttonEvent.name == rt_btn.ButtonName.F3:
                model_params["show_signs"] = not model_params["show_signs"]
            elif buttonEvent.name == rt_btn.ButtonName.O:
                break
 
 
