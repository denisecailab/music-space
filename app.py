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

pn.extension(
    "plotly",
    theme="dark",
    design="material",
    sizing_mode="stretch_width",
    template="fast",
    notifications=True,
)
pn.state.template.param.update(site="CaiShumanGroup", title="Lab Music Space")
notif = pn.state.notifications  # https://github.com/holoviz/panel/issues/5488


class MusicSpace:
    def __init__(self) -> None:
        self.data = None
        self.wgt_pw = pn.widgets.PasswordInput(
            name="Password", placeholder="Enter password (Enter to confirm)"
        )
        self.wgt_pw.param.watch(self.cb_pw, "value")
        self.plot_proj = None
        self.layout = pn.Column("Lab Music Space", self.wgt_pw, self.plot_proj)

    def serve(self) -> pn.Column:
        return self.layout.servable()

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

    def update_proj_plot(self, theme=None):
        pca = PCA(n_components=3, whiten=True)
        pcs = pca.fit_transform(self.data[self.feats])
        for i in range(3):
            self.data["pc{}".format(i)] = pcs[:, i]
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
        if self.plot_proj is None:
            self.plot_proj = pn.pane.Plotly(fig)
            self.layout[2] = self.plot_proj
        else:
            self.plot_proj.object = fig

    def cb_pw(self, evt):
        pw = evt.new
        if pw:
            try:
                self.decrypt_data(pw)
            except:
                notif.error("Invalid password")
                return
            try:
                self.setup_spotify()
                notif.success("Authentication success")
            except:
                notif.error("Authentication failed, check your password")
                return
            self.update_proj_plot()


# %% serve app
ms = MusicSpace()
ms.serve()
