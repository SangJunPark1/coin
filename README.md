# 코인 현물 자동거래 MVP

이 프로젝트는 코인 현물 자동거래 시스템을 배우기 위한 보수적인 MVP입니다.
현재 버전은 **실거래를 하지 않는 모의거래 전용**입니다.

## 현재 가능한 것

- 업비트 공개 시세 또는 샘플 데이터로 가격을 가져옵니다.
- 단순 이동평균 전략으로 매수, 매도, 대기 신호를 만듭니다.
- 실제 주문 대신 모의 체결을 기록합니다.
- 하루 손실 한도, 하루 수익 목표, 신규 진입 횟수 제한을 적용합니다.
- 거래 기록과 판단 로그를 저장합니다.
- HTML 리포트로 결과를 보기 좋게 확인합니다.

## 안전 기본값

- 기본 모드는 `paper`입니다.
- 실거래 주문 기능은 아직 구현하지 않았습니다.
- 선물, 마진, 레버리지, 숏, 고빈도 주문 반복, 허수 주문, 다계정 거래는 범위에서 제외합니다.
- 봇이 거래하지 않는 것도 정상적인 판단입니다.

## VSCode에서 실행하기

VSCode에서 이 폴더를 연 뒤 다음 메뉴를 사용하세요.

```text
Terminal > Run Task...
```

추천 실행 순서:

1. `코인 MVP: 테스트 실행`
2. `코인 MVP: 샘플 실행 + 리포트`
3. `reports/latest_report.html` 열기
4. 필요하면 `코인 MVP: 업비트 공개시세 + 리포트` 실행
5. 휴대폰에서 보려면 `코인 MVP: 휴대폰 리포트 서버 실행` 실행
6. 같은 Wi-Fi가 아니어도 보려면 `코인 MVP: 외부 모바일 링크 실행` 실행

`latest_report.html`은 코드 화면으로 열릴 수 있습니다. 보기 좋게 보려면 브라우저로 열거나 VSCode 확장 `Live Preview`를 사용하세요.

## 휴대폰에서 리포트 보기

PC와 휴대폰이 같은 Wi-Fi에 연결되어 있으면 휴대폰 브라우저에서도 리포트를 볼 수 있습니다.

VSCode에서 아래 작업을 실행하세요.

```text
Terminal > Run Task... > 코인 MVP: 휴대폰 리포트 서버 실행
```

터미널에 이런 주소가 표시됩니다.

```text
휴대폰에서 보기: http://내_PC_IP:8765/latest_report.html
```

그 주소를 휴대폰 브라우저에 입력하면 됩니다.

주의:

- 같은 Wi-Fi가 아니면 접속되지 않을 수 있습니다.
- Windows 방화벽이 Python 접속 허용을 물어보면 개인 네트워크에서 허용하세요.
- 서버를 끄려면 VSCode 터미널에서 `Ctrl+C`를 누르세요.

## 외부 모바일 링크로 보기

같은 Wi-Fi가 아니어도 보려면 Cloudflare Tunnel을 사용합니다.

먼저 한 번만 설치하세요.

```powershell
winget install --id Cloudflare.cloudflared
```

그 다음 VSCode에서 아래 작업을 실행하세요.

```text
Terminal > Run Task... > 코인 MVP: 외부 모바일 링크 실행
```

터미널에 `https://...trycloudflare.com` 형태의 주소가 표시됩니다.
휴대폰에서는 아래 형태로 접속하면 됩니다.

```text
https://표시된주소/latest_report.html
```

주의:

- 이 링크는 터미널이 켜져 있는 동안만 작동합니다.
- 링크를 아는 사람은 리포트를 볼 수 있으므로 공유하지 마세요.
- 실거래 API 키나 개인정보를 리포트에 표시하면 안 됩니다.
- 종료하려면 해당 터미널에서 `Ctrl+C`를 누르세요.

## PowerShell에서 직접 실행하기

샘플 데이터로 실행:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sample.ps1
```

업비트 공개 시세로 실행:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_upbit.ps1
```

테스트 실행:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
```

리포트만 다시 생성:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate_report.ps1
```

## 주요 파일

- `config.example.json`: 전략과 리스크 설정
- `coin_mvp/strategy.py`: 매수/매도 신호 생성
- `coin_mvp/risk.py`: 손실 제한, 수익 목표, 진입 제한
- `coin_mvp/broker.py`: 모의 체결 처리
- `coin_mvp/report.py`: HTML 리포트 생성
- `data/trades.csv`: 거래 기록
- `logs/events.jsonl`: 판단 로그
- `reports/latest_report.html`: 보기 좋은 리포트

## 설정에서 자주 바꿀 값

`config.example.json`에서 아래 값을 조정할 수 있습니다.

- `market`: 거래 관찰 대상. 예: `KRW-BTC`
- `starting_cash`: 모의거래 시작 금액
- `daily_profit_target_pct`: 하루 수익 목표
- `daily_loss_limit_pct`: 하루 손실 한도
- `position_fraction`: 신규 진입 때 사용할 현금 비중
- `max_entries_per_day`: 하루 신규 진입 횟수 제한
- `take_profit_pct`: 포지션 익절 기준
- `stop_loss_pct`: 포지션 손절 기준

## Codex에게 요청하기 좋은 작업

- "리포트에 누적 손익 그래프를 추가해줘."
- "최근 거래 승률과 평균 손익을 더 자세히 보여줘."
- "전략을 변동성 돌파 방식으로 하나 더 추가해줘."
- "업비트 공개 시세로 1시간 관찰한 로그를 분석해줘."
- "손실이 2번 연속이면 그날 신규 진입을 막도록 바꿔줘."

## 주의

이 프로젝트는 교육용 엔지니어링 예제입니다. 특정 코인 매수나 매도를 추천하지 않습니다.
실거래 API를 붙이기 전에는 최소 2-4주 이상 모의거래로 로그를 쌓고, 손실 제한과 중지 조건이 제대로 작동하는지 먼저 확인하세요.
