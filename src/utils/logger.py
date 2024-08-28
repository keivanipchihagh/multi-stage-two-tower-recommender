from colorama import Fore
from pytz import timezone
from datetime import datetime
from colorama import init as colorama_init

colorama_init()
TZ = timezone('Asia/Tehran')    # Timezone


def __now() -> str:
    """ Returns the current datetime as string """
    return datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')


def debug(message: str) -> None:
    """ Prints a message as debug """
    print(f"[{Fore.MAGENTA}DEBUG{Fore.BLUE} - {Fore.BLACK}{__now()}{Fore.RESET}]\t{message}")


def info(message: str) -> None:
    """ Prints a message as info """
    print(f"[{Fore.BLUE}INFO{Fore.BLUE} - {Fore.BLACK}{__now()}{Fore.RESET}]\t{message}")


def success(message: str) -> None:
    """ Prints a message as success """
    print(f"[{Fore.GREEN}SUCCESS{Fore.RESET} - {Fore.BLACK}{__now()}{Fore.RESET}]\t{message}")


def error(message: str) -> None:
    """ Prints a message as error """
    print(f"[{Fore.RED}ERROR{Fore.RESET} - {Fore.BLACK}{__now()}{Fore.RESET}]\t{message}")


def warning(message: str) -> None:
    """ Prints a message as warning """
    print(f"[{Fore.YELLOW}WARNING{Fore.RESET} - {Fore.BLACK}{__now()}{Fore.RESET}]\t{message}")
