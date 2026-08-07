Option Explicit

Dim shell, filesystem, root, pythonw, agent, plugin, telemetry, command

Set shell = CreateObject("WScript.Shell")
Set filesystem = CreateObject("Scripting.FileSystemObject")

root = filesystem.GetParentFolderName(WScript.ScriptFullName)
pythonw = shell.ExpandEnvironmentStrings( _
    "%LOCALAPPDATA%\Microsoft\WindowsApps\pythonw.exe")
agent = filesystem.BuildPath(root, "scripts\run_th105_agent.py")
plugin = filesystem.BuildPath(root, "scripts\th105\policies\adaptive.py")
telemetry = filesystem.BuildPath(root, "runtime\th105_live.jsonl")

If Not filesystem.FileExists(pythonw) Then
    WScript.Quit 2
End If

command = Quote(pythonw) _
    & " " & Quote(agent) _
    & " auto-arcade --launch" _
    & " --p1-character patchouli" _
    & " --difficulty lunatic" _
    & " --continuous" _
    & " --playstyle aggressive" _
    & " --exploration-rate 0.08" _
    & " --policy-plugin " & Quote(plugin) _
    & " --telemetry-path " & Quote(telemetry)

' Window style 0 plus the GUI-subsystem pythonw executable leaves no controller
' console to steal focus or to terminate the learner when a terminal is closed.
shell.Run command, 0, False
WScript.Quit 0

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
