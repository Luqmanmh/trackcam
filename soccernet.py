import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="C:/Users/Luqman/OneDrive/Documents/luqman/Proj/cam")

mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_1080p.mkv", "2_1080p.mkv"], split=["train","valid","test","challenge"])