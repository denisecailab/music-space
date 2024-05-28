# %% import and definition
import base64
from io import StringIO

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import spotipy
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sklearn.decomposition import PCA
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

FEATS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]
KEY_SALT = b"\x89 \xa9\xf4\xe1\xbe\x84\x01'ym\xf0k\xcf\x7f{"
APP_ID = b"gAAAAABmVWcj69yZexqsTUWDOri8TQsvqyO36J5Qbm5G1rJEJQ3ePqRtIL9CzNQDQpRsfJa6iCWFjTUHpdf2g37qPWn7h00Nn7UYKhIkDmQ6NaHrwaLMZd-s7-Nx8Z2MebFe6ilK_yBy"
APP_SECRET = b"gAAAAABmVWcjpgtCHRP6qeZfnu_y-6-6BMz3L9pOWDBzpX2Zq1ng5pIXall8Z6GmTlW4BMGHOkP3CUSVvvVNAU0pEnS0arve8dWOLbcczgJbHMnjiKXQGjDqvA1SmV27UzbXTgtLK79V"
DATA = b"gAAAAABmVWcjVS0wbucdFxW4YXkPQv6z6afksP3CQ5jCSPgXrrjF-GG42szszvnE-HV5DMhHIoHL_OpqRkvxq1yzk_WdWTMQeZ7X10SELeAOx7unoQkdw901e0TYkoj4f-yp9tazuW0JCiOo8QK_MKI4VgCviZ-gFLPf8V-AczckQg39qScaUZ0JoRV8RkJieCrKSuzLcqAzEgj-sZvrTC81yLsSLzipHKu_bx2u6ef-WPCZp0p7jDXhTbvy_Xm8UYvFOEArtNWHzBE3NXR21uk1Zrx2LvIpm2oUPdSqVr0JOWZaToJMnK6SfZEbeKvH7dD0XK2KoBcJvi-xhCunwWBoy4hYISn_ld75I8f4DmBnHANYdd-1by1C-AIqfhLL3aZGbGVRbNIOFuyXfr0dF5IixkO2Y9evIAdhtIZTSXLvrJScR9185ar98o19r5ii_qkux6laFBIRlU-6QykTSZ6dWY5_xMtsyoWrBYBkruhzZnEcA5RVSW3isox0DdNnITRfeOpkp9D5RxsPZEp5r2plZIKbJpjCufaNBga_31ovoid9dSAJuSXvGf6Q_JjikxSMYYu78qFWOPm_DYVhgr_sWMs2bkgc-cWHaLYHxv--h_QIGdIe0K4dlqX0gZ1KvJQZ6a7DUf7XAv5bhcDtrc1ZFlVRYYPl45C9HrfrRG7EyIB3RxT88O38Mn99a3WVGpl8qJKDDu1WTlLeul-BaAywncf51ehzL49tL8Qhbio4kQVjYNjEOJJbOemX4M5aJb17hk3X7FeiM1AU6S2-z7UoRuX72BXORJdZIfgrnN9wp4YfLu-c9gjSpkBbjti90ce1NmsZT7CUJQXNbqGOO5iKBMIUarCmgn0pj3rxWMWp4PzzQm0F_gbLVdNzvrkQ6ddUr8MHBuiaib1VTeIEuYbId-m1HPyEduM5MGqSRJMo5uWEBOrNwibm5uuGaaWSOibC2Rp-kMHqQJRUKGF2_5svzYuKrx7fCPWV1xDDyL2SMzvYIfikH-QJRqWUs6gUoVJdkHRCUEw6CqIRGwkxYye_XlVj-LtehW7uAXJfAYdv90tyAHwwytwtLiZRXQhyce1WHS7FfOmgvasyZs754xH0-pcDgOLU2VeCigS4WuBknIrpJXyeGqrJxKpMxAfmm4SmFy686WUavj7D1dfaZ4xr91ryBLoOk4VusMDlitGBpEkxfiIhA5_8MKbfnwfSR2WDs-Ygye-ktY6c0kTV92HOyG89h5xS4MjoAaKuyQLBKS-hBIFomDx_z_PuRA4oLhZ6Ax1Uin2MOLeIPD86Hbh69JCK1w-HW1QAHnXPfd8fEikXpOzekOG2OfM8Wa7l2OPnBfz4N1c9j7bkP_NceCOsdxu9t2aSfMaTYLxStjJJb6yD0K09yrnfFs0eWzITHfFopW7XHXOvkV4F-yBCrNY3TtUCN8-otaDhAmQPy9anwiibljJiMQzxEt3djn2vxsyUwKXa2wErZ1KFRKkI13onWHYi1325zSluTWZH2NAMhIPuHAUF5aliQjtEMqMvIxr2rFmwhB2U5wEZPjNUuwhZueiEsfSgWrS7eSPXMwXueRkkT3Tw25NxraBa3yT0I60rn7eD6eNU_ibPngPtQAMVyfiRRqeA7flN8QB_ZKfPWybR8KaPZ2GgdqCxfmIBvgsuAfsHer0xwoym81qH0fgYJ0L_SUr-IGhFrm0PYn956OD6POrpIAl4yk57S6qBCCEdv8_NB96iUvVN6YNoETKVKa588XZaxJplMW6-x38IoZV02sbJGwAqndY3THXVCEdxawGdRlsNttlT1YQbIzZroE77nUkhg77n6Ew5VpNQ5eKh_6AQLoVCiypXN4c_FKpRgfm9IMcjgGvzdqWNa-TKyc1rQfH2WIq61CMnBRhPGIFmTWr_NnWFQM0MIUkW_n6m-oz-GV7hTenE4LYdNvb4SKcxuUU-vA_bwhhOv3YOzwfzdivWsv5YCBY_Nu9ARbg9ZN9vAbq3P0QmCWdj83GLw_xEPPeURT0jY0A15-RyYoeDX65_hBGkSr0Y-EPVIzj03j7fQdy00LYrVRCkeYX2iyZd-NuHrtiMuKXE8s3wXKOcrTqZZLMwrw3bEzt_9ySLQI7HopRuA3WWeUB_xO-iYJHOfW9XqqnOQtlk-6vxAr-Jtl7lBszf69TDTSXlJpFpaoWgwjSkHMzUAUQgEkdA1VqHW1j8mj5sohb7PKBAACM082cSqJyP6v-ikuTfZZwZEXvKNE0CrkhAQL2HSMDXVi_RpEoceP1EElpgSdugP6jveoNWsFQKZpmQGjD06mB2iJ-7WHEvsdfc6lN468RUg1Xnm32_52SlWpwC95gDV9z0OvLwwHvlx4P9bkMmgXLSyrBAKlWNkjcdzBwKe-BISy2JeeXDuAFOX9SNPxZYcoMxH2dsToEgV6FotydbWaYbqSoCw_yJcHzdb9ZsNIBhsi8KA2gMele-WM-0LEdew8Z6tM6C01zxPpz5O5W8l4eeL40MliaaWTHy7Ez_4zy0e0euB8B4Yy4feytooQFAKzO0_TvO0yLK0mqZ4HZKGfBIWm2OkrnpKuXyLnXqKrz5eFpmQyQoBG3cdCPjuRHCxIZGQOBA13MXUqfj-hTbqYa_Hhvzy8vh1rfMEIN9bF8mMNrHH01Xc2Xo456kBjzqoojCzGx9mZ2hsygzRG2Vs8WX009O3anLkTOU9z1IoqNwMk0nWbjvEq2xWl2r-CuYqMfvXRXvQ15O1xZTGCT5ZtSBRDDqKZXO-SMN6ABG3HX8imaiA_S2mqyUEjXTAq1eYn9ZCk5eVhGoTrGdcSRZSwNk12an54k7F74cnREj6zH1rzul8So4w67mK2AgduURJ746oD0mkv3VTmYCH9wVbTwcGirMO_ZKLZnoEMYcz183BpjgwIwZQnb7m9hcjBO3Ct7mB7wHTyuOFh0cd85epWQxksCud-7ensSGFPmBA3kV5hKu3LisMrM3v1U-J-bf8jJ8KWk_5d5LvX4AinluIakCx6JdZWFWdEJawkhMeq9s_WRqP_TmqdKvpSdDiivAP9M8IvrDEGQYrZXmhxvaZKRocl9_ZR1MtKuSAN9f7nd-au2I3Gf5Z0mx18h0NvUfLiazoWyw2UU7hHU6t_F1NF95XSxN92zh1CCU9Flzi0Xj77HJYZexVctMPCb3LNRQY78ap3WTqOynSoDSeYGU-k8ps3GEN1DZTaTJJMyqOIsopeJM0XiJ4eDPBDugS6LBphy1alm_i3Q702XYLEafCZv9oDDfRnh9rcwZK8pZJLzvZYe_4k6lrjjRQssAUVyiqQ0bqdN7xknvcMnf7tnvGbfHJpBBFqHGUlb0zXUuaonIcSQXnQAylfEigUDsYlJvirmgS_CUcigJ83_WH1ebaqfAAYxTDv2UgBWTh_uY6pLA5fBQLZJrJg=="

pn.extension("plotly", notifications=True)


class MusicSpace:
    def __init__(self) -> None:
        # build app
        self.template = pn.template.MaterialTemplate(
            theme="dark", site="CaiShumanGroup", title="Lab Music Space"
        )
        # https://github.com/holoviz/panel/issues/5488
        self.notif = pn.state.notifications
        wgt_pw = pn.widgets.PasswordInput(
            name="Password", placeholder="Press <Enter> to confirm"
        )
        wgt_pw.param.watch(self.cb_pw, "value")
        modal_btn = pn.widgets.Button(name="Enter Password")
        modal_btn.on_click(self.cb_modal)
        self.plot_proj = pn.pane.Plotly()
        self.layout_main = pn.Column(modal_btn, sizing_mode="stretch_width")
        self.layout_modal = pn.Column("Hint: c#1", wgt_pw)
        self.template.main.append(self.layout_main)
        self.template.modal.append(self.layout_modal)
        # init data
        self.auth_success = False
        self.data = None
        self.model = None

    def serve(self) -> pn.Column:
        return self.template.servable()

    def decrypt_data(self, pw) -> None:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=KEY_SALT,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(pw.encode("utf-8")))
        fernet = Fernet(key)
        self.app_id = fernet.decrypt(APP_ID).decode("utf-8")
        self.app_secret = fernet.decrypt(APP_SECRET).decode("utf-8")
        self.data_raw = fernet.decrypt(DATA).decode("utf-8")

    def setup_spotify(self) -> None:
        auth = SpotifyClientCredentials(
            client_id=self.app_id, client_secret=self.app_secret
        )
        self.feats = FEATS
        self.sp = spotipy.Spotify(client_credentials_manager=auth)
        self.data = pd.read_csv(StringIO(self.data_raw))

    def populate_feats(self):
        uris = self.data["uri"]
        tracks = self.sp.tracks(uris)["tracks"]
        feats = self.sp.audio_features(uris)
        self.data["name"] = [t["name"] for t in tracks]
        self.data["id"] = [t["id"] for t in tracks]
        self.data["artist"] = [t["artists"][0]["name"] for t in tracks]
        for fn in self.feats:
            self.data[fn] = [f[fn] for f in feats]
        self.data["new"] = False

    def add_entry(self, member, uri):
        try:
            track = self.sp.track(uri)
        except SpotifyException:
            self.notif.error("Invalid Spotify URI")
            return
        if not (track["id"] in self.data["id"].to_list()):
            feats = self.sp.audio_features(uri)[0]
            feats = {fn: feats[fn] for fn in self.feats}
            dat = {
                "member": member,
                "uri": uri,
                "name": track["name"],
                "id": track["id"],
                "artist": track["artists"][0]["name"],
                "new": True,
            }
            dat.update(feats)
            pcs = self.model.transform(
                np.array(list(feats.values())).reshape((1, -1))
            ).squeeze()
            dat.update({"pc{}".format(i): p for i, p in enumerate(pcs)})
            self.data = pd.concat([self.data, pd.DataFrame([dat])], ignore_index=True)

    def update_main(self):
        if self.auth_success:
            wgt_title = pn.pane.Alert("Welcome to Lab Music Space!", alert_type="dark")
            wgt_info = pn.pane.Alert(
                """
                - Input the name and favorite song of new lab member to show in music space
                - Hover over individual points to see more info
                """,
                alert_type="info",
            )
            self.wgt_member = pn.widgets.TextInput(name="New Member")
            self.wgt_link = pn.widgets.TextInput(name="Spotify Link")
            wgt_add = pn.widgets.Button(name="Add Member")
            wgt_add.on_click(self.cb_add_member)
            self.layout_main.clear()
            self.layout_main.extend(
                [
                    wgt_title,
                    wgt_info,
                    pn.Row(self.wgt_member, self.wgt_link, wgt_add),
                    self.plot_proj,
                ]
            )
            self.template.close_modal()

    def update_annotation(self):
        new_dat = self.data[self.data["new"]]
        for idx, row in new_dat.iterrows():
            self.plot_proj.object.add_trace(
                px.scatter_3d(
                    row.to_frame().T,
                    x="pc0",
                    y="pc1",
                    z="pc2",
                    color="member",
                    hover_data=["member", "artist", "name"],
                ).data[0]
            )
            old_annot = self.plot_proj.object.layout.scene.annotations
            self.plot_proj.object.update_layout(
                scene={
                    "annotations": list(old_annot)
                    + [
                        {
                            "x": row["pc0"],
                            "y": row["pc1"],
                            "z": row["pc2"],
                            "text": row["member"],
                        }
                    ]
                }
            )
            self.data.loc[idx, "new"] = False

    def update_model(self):
        self.model = PCA(n_components=3, whiten=True)
        pcs = self.model.fit_transform(self.data[self.feats])
        for i in range(3):
            self.data["pc{}".format(i)] = pcs[:, i]

    def update_proj_plot(self, theme=None):
        theme = "plotly" if theme == "light" else "plotly_dark"
        fig = px.scatter_3d(
            self.data,
            x="pc0",
            y="pc1",
            z="pc2",
            color="lab",
            template=theme,
            hover_data=["member", "artist", "name"],
        )
        fig.layout.autosize = True
        self.plot_proj.object = fig

    def cb_modal(self, evt):
        self.template.open_modal()

    def cb_pw(self, evt):
        pw = evt.new
        if pw:
            try:
                self.decrypt_data(pw)
            except:
                self.notif.error("Invalid password")
                return
            try:
                self.setup_spotify()
                self.auth_success = True
                self.notif.success("Authentication success")
            except:
                self.notif.error("Authentication failed, check your password")
                return
            try:
                self.populate_feats()
            except:
                self.notif.error("Data corrupted, check uri")
                return
            self.update_model()
            self.update_proj_plot()
            self.update_main()

    def cb_add_member(self, evt):
        self.add_entry(self.wgt_member.value_input, self.wgt_link.value_input)
        self.update_annotation()


# %% serve app
ms = MusicSpace()
ms.serve()
