from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .messages import MotionCommand


@dataclass
class SerialConfig:
    """SerialConfig: UART(시리얼) 전송 설정.

    UART(serial): PC <-> MCU(예: Arduino) 사이에 텍스트/바이너리로 데이터를 보내는 통신.
    예시: "s7l100r100\n" 같은 문자열을 보냄.
    """

    port: Optional[str] = None
    baud: int = 115200
    timeout: float = 0.0


class SerialSender:
    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self.ser = None
        if cfg.port:
            try:
                import serial
            except Exception as e:
                raise RuntimeError("pyserial 미설치 또는 import 실패. 'pip install pyserial' 후 재실행하세요.") from e
            self.ser = serial.Serial(cfg.port, cfg.baud, timeout=cfg.timeout)

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    @staticmethod
    def encode_command(cmd: MotionCommand) -> str:
        # ROS2 protocol_convert_func_lib.py와 동일 형식
        # s{steering}v{speed}\n (l, r 대신 v 사용)
        return f"s{cmd.steering}v{cmd.left_speed}\n"

    def send(self, cmd: MotionCommand) -> None:
        if self.ser is None:
            return
        msg = self.encode_command(cmd)
        # print(f"[UART] {msg.strip()}")  # 디버깅용 출력 (속도 저하 원인이므로 주석 처리)
        self.ser.write(msg.encode("ascii", errors="ignore"))

    def read(self) -> Optional[str]:
        """시리얼 포트에서 들어온 데이터를 모두 읽어 반환합니다. 데이터가 없으면 None."""
        if self.ser is None:
            return None
        
        try:
            # timeout=0(또는 0.1)일 때 read(size)는 가능한 만큼 읽고 반환
            data = self.ser.read(1024) 
            if data and len(data) > 0:
                # 디버깅: 원본 바이트가 들어오는지 확인하려면 아래 주석 해제
                # print(f"[RAW] {data}")
                return data.decode("ascii", errors="ignore").strip()
        except Exception:
            pass
        return None
