"""Adds a read-only GET /frame/<camera>.jpg to Dream station API (the controller latest frame), for our Jev judge.
Loaded only when PYTHONPATH includes this folder; Dream files are untouched."""
try:
    import re as _re
    import cv2 as _cv2
    from robots import station_api as _sa

    _orig_get = _sa._Handler.do_GET

    def _do_get(self):
        if self.path.split("?", 1)[0] == "/arms":  # each gripper: measured opening, commanded opening, effort
            out = {}
            for name, st in self.server.api.controller.arm_states().items():
                out[name] = {"gripper": float(st.gripper),
                             "commanded_gripper": None if st.commanded_gripper is None else float(st.commanded_gripper),
                             "gripper_effort": None if st.gripper_effort is None else float(st.gripper_effort)}
            return self._reply(200, out)
        m = _re.match(r"^/frame/([A-Za-z0-9_]+)\.jpg$", self.path.split("?", 1)[0])
        if not m:
            return _orig_get(self)
        frame = self.server.api.controller.frames.get(m.group(1))
        if frame is None:
            return self._reply(404, {"error": f"no frame from camera {m.group(1)!r}"})
        ok, buf = _cv2.imencode(".jpg", _cv2.cvtColor(frame.data, _cv2.COLOR_RGB2BGR), [_cv2.IMWRITE_JPEG_QUALITY, 85])
        body = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    _sa._Handler.do_GET = _do_get
except Exception as _e:  # never stop Dream station over this
    import sys as _sys
    print(f"pah_judge: frame route not added: {_e}", file=_sys.stderr)
