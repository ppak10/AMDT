from amdt import Simulator

s = Simulator()
gcode_commands = s.load_gcode_commands("3DBenchy.gcode")
gcode_segments = s.gcode_commands_to_segments(gcode_commands)
