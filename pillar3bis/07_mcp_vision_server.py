"""
Step 7 - your vision tools as an MCP server.

Same idea as pillar 3 step 7, with the CV specialists: once this runs, any MCP client can
check image quality, find anomalies, run YOLO or ask the VLM about images on your disk.
For example from VS Code agent mode: "check the quality of every image in the plates folder
and inspect the ones with anomalies".

Test it with the pillar 3 client (it accepts another server as argument):
    cd ..\\pillar3
    python 07_mcp_client_agent.py ..\\pillar3bis\\07_mcp_vision_server.py

Images can be given as names from list_images (data/plates) or as absolute paths, so it also
works on your own photos.
"""
import functools

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import vision_tools as vt

mcp = MCPServer(
    name="vision-inspection",
    instructions="Computer vision tools for visual quality control: image quality, anomaly detection "
                 "against a golden sample, crops, YOLO object detection, CLIP and a vision-language model.",
)


def expose(t):
    fn = t.fn  # the plain function behind our pillar 3 Tool wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # pass our readable errors through (see pillar 3 step 7)
            raise ToolError(str(e)) from e

    # crop_region writes a file, ask_vlm calls an external API: not "read only"
    read_only = t.name not in ("crop_region", "ask_vlm")
    mcp.tool(annotations=ToolAnnotations(read_only_hint=read_only))(wrapper)


for t in vt.TOOLS:
    expose(t)


if __name__ == "__main__":
    mcp.run(transport="stdio")
