from tools import Realtime
from settings import Settings
from model import load

model = load()

settings = Settings()

realtime = Realtime(settings,model)

call = realtime.callback

realtime.run(call)



