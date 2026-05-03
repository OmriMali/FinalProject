from tqdm import tqdm
import sys


class ConsoleReporter():

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.active = False
        self.bars = {}

    def _fmt(self, msg: str):
        if self.prefix:
            return f"[{self.prefix}] {msg}"
        return msg
    
    # session control

    def start(self, title: str):
        self.active = True
        print(self._fmt(f"START: {title}"))

    def end(self, message: str = ""):
        for bar in self.bars.values():
            bar.close()

        self.bars.clear()
        self.session_active = False

        print(self._fmt(f"DONE: {message}"))

    # messages

    def message(self, text: str):
        print(self._fmt(text))


    # progress bars

    def create_bar(self, name: str, total: int = 100, desc: str = ""):
        bar = tqdm(total=total, desc= desc or name, file=sys.stdout)
        self.bars[name] = bar
        return name
    
    def update(self, name: str, progress: float, text: str = ""):
        if name not in self.bars:
            return

        bar = self.bars[name]
        bar.n = int(progress * bar.total)

        if text:
            bar.set_description(text)

        bar.refresh()

