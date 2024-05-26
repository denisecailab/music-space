# %% import and definition
import base64
from io import StringIO

import pandas as pd
import panel as pn
import plotly.express as px
import spotipy
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sklearn.decomposition import PCA
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
KEY_SALT = b"\xf3j\x10\x1a\x8f\xd8\xb6\xe7:c\xa0\xb9\xd7:\x9a\xdb"
APP_ID = b"gAAAAABmU2m_uhvOc7M0xlVdGo4ylOwwpIFWqnqdM0Thmpt2MakWrFUlBnuxM2IOS4_WL-hyhJ468MRFyFIJsi-xmIBHCoCD0k3LNpkEkHwqluywgXU5Wz9zrY92VHHuolEne4yXiMz_"
APP_SECRET = b"gAAAAABmU2m_v37-FNEm-gdvIYntNgXTz4YVvPaLGTl4G3ntSnNMt2xdnK7tbH8_qQeulrbrP1q9bIT0kTl4baLFhjDW0DPAMlH8ezDD22yKam6RRss25i8_vI2V_Hj8nGeHOKctTmjS"
DATA = b"gAAAAABmU2m_pqhLmAPTjjzP7vIEGTiZjwvR0bMh90ppn5-K71mhTbrsM6zOphOmiH8TcKMQjJ6YAe02DTdl6rOO-HGD6kn_k1NqIWtns-A27iRAe2QwfdFWkOCQjtHViNRnI2Ax7h9RmLSQ_9IcSPdLrfAldqKCOSOM6LM59SedHPPq4QVhYGe5fsqb71ns8DLyjaUfpDum2phkfY6ncDEjXUbMw3j1S23shIii5bh-4Mc4xeSwnL9hT7gdPU_ckU4alUKXpqzpylfPEejFAlCcFx2gLJ_V1haxA_WskU7lHnRAPFCFY_Ki-ar2zorQlz0e8E7npZFZVvnIQojQ-8QyNT94rD0JtmBj4n822VXSiV5ytboijiymTETj1t6-JH1heomY7rUiGSF7p56NTiFfyg_Wehszaw=="

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
        self.layout_main = pn.Column(modal_btn)
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
        self.populate_feats()

    def populate_feats(self):
        uris = self.data["uri"]
        tracks = self.sp.tracks(uris)["tracks"]
        feats = self.sp.audio_features(uris)
        self.data["name"] = [t["name"] for t in tracks]
        self.data["artist"] = [t["artists"][0]["name"] for t in tracks]
        for fn in self.feats:
            self.data[fn] = [f[fn] for f in feats]

    def update_main(self):
        if self.auth_success:
            self.layout_main[0] = pn.panel("Welcome to Lab Music Space")
            self.layout_main.append(self.plot_proj)
            self.template.close_modal()

    def update_model(self):
        pca = PCA(n_components=3, whiten=True)
        pcs = pca.fit_transform(self.data[self.feats])
        for i in range(3):
            self.data["pc{}".format(i)] = pcs[:, i]

    def update_proj_plot(self, theme=None):
        theme = "plotly" if theme == "light" else "plotly_dark"
        fig = px.scatter_3d(
            self.data,
            x="pc0",
            y="pc1",
            z="pc2",
            color="member",
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
            self.update_model()
            self.update_proj_plot()
            self.update_main()


# %% serve app
ms = MusicSpace()
ms.serve()
