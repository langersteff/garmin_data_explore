import vlc

# Create Player Object and set MRL (your_video.mp4)
player = vlc.MediaPlayer('/Users/stefanlanger/projects/artificial-intelligence/fittest/data/warngaus2_1.mp4')
player.play()
time.sleep(10)
player.stop()