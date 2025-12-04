import argparse

from win_send import send_text_to_pid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Envoyer une ligne dans la fenêtre Codex via son PID.")
    parser.add_argument("text", help="Texte à envoyer.")
    parser.add_argument("--pid", type=int, required=True, help="PID de la fenêtre Codex.")
    parser.add_argument("--no-submit", action="store_true", help="Ne pas envoyer Enter après le texte.")
    parser.add_argument("--delay", type=float, default=0.02, help="Délai entre touches (secondes).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok = send_text_to_pid(args.pid, args.text, submit=not args.no_submit, delay=args.delay)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
