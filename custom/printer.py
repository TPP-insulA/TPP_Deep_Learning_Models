class ForegroundColours:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    ORANGE = "\033[38;5;208m"
    LIGHT_GRAY = "\033[38;5;250m"
    LIGHT_RED = "\033[38;5;196m"
    LIGHT_GREEN = "\033[38;5;82m"
    LIGHT_YELLOW = "\033[38;5;226m"
    LIGHT_BLUE = "\033[38;5;39m"
    LIGHT_MAGENTA = "\033[38;5;201m"
    LIGHT_CYAN = "\033[38;5;51m"
    LIGHT_WHITE = "\033[38;5;231m"
    LIGHT_ORANGE = "\033[38;5;214m"
    DARK_GRAY = "\033[38;5;235m"
    DARK_RED = "\033[38;5;124m"
    DARK_GREEN = "\033[38;5;28m"
    DARK_YELLOW = "\033[38;5;220m"
    DARK_BLUE = "\033[38;5;21m"
    DARK_MAGENTA = "\033[38;5;125m"
    DARK_CYAN = "\033[38;5;51m"
    DARK_WHITE = "\033[38;5;231m"
    DARK_ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"

class BackgroundColours:
    BLACK = "\033[40m"
    RED = "\033[41m"
    GREEN = "\033[42m"
    YELLOW = "\033[43m"
    BLUE = "\033[44m"
    MAGENTA = "\033[45m"
    CYAN = "\033[46m"
    WHITE = "\033[47m"
    ORANGE = "\033[48;5;208m"
    RESET = "\033[0m"
    
class Styles:
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    REVERSED = "\033[7m"
    RESET = "\033[0m"
    STRIKETHROUGH = "\033[9m"
    HIDDEN = "\033[8m"
    ITALIC = "\033[3m"
    DIM = "\033[2m"
    BLINK = "\033[5m"
    INVERSE = "\033[7m"
    CROSSED_OUT = "\033[9m"
    NEGATIVE = "\033[7m"
    FRAMED = "\033[51m"
    ENCIRCLED = "\033[52m"
    OVERLINED = "\033[53m"
    DOUBLE_UNDERLINE = "\033[21m"
    SLOW_BLINK = "\033[5;2m"
    FAST_BLINK = "\033[5;1m"
    CONCEALED = "\033[8m"

def coloured(text, colour=None, background=None, style=None):
    """
    Colorea el texto con el color y estilo especificados.
    
    Args:
        text (str): Texto a colorear.
        colour (str): Color del texto. Opciones: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
        background (str): Color de fondo. Opciones: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
        style (str or list): Estilo del texto o lista de estilos. Opciones: 'bold', 'underline', 'reversed', etc.
        
    Returns:
        str: Texto coloreado.
    """
    if colour:
        text = f"{getattr(ForegroundColours, colour.upper(), '')}{text}"
    if background:
        text = f"{getattr(BackgroundColours, background.upper(), '')}{text}"
    
    if style:
        # Handle both string and list inputs for backward compatibility
        if isinstance(style, str):
            style_codes = getattr(Styles, style.upper(), '')
            text = f"{style_codes}{text}"
        else:
            # Apply each style in the list
            for s in style:
                style_codes = getattr(Styles, s.upper(), '')
                text = f"{style_codes}{text}"
    
    return f"{text}{Styles.RESET}"

def cprint(text, colour=None, background=None, style=None):
    """
    Imprime el texto coloreado en la consola.
    
    Args:
        text (str): Texto a imprimir.
        colour (str): Color del texto. Opciones: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
        background (str): Color de fondo. Opciones: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
        style (str or list): Estilo del texto. Opciones: 'bold', 'underline', 'reversed'.
    """
    print(coloured(text, colour=colour, background=background, style=style))
    
def print_error(text):
    """
    Imprime un mensaje de error en rojo.
    
    Args:
        text (str): Mensaje de error a imprimir.
    """
    print(coloured("ERROR", colour='white', background='red', style='bold') + " " + coloured(text, colour='red', style='bold'))

def print_warning(text):
    """
    Imprime un mensaje de advertencia en amarillo.
    
    Args:
        text (str): Mensaje de advertencia a imprimir.
    """
    print(coloured("WARNING", colour='black', background='yellow', style='bold') + " " + coloured(text, colour='yellow', style='bold'))

def print_success(text):
    """
    Imprime un mensaje de éxito en verde.
    
    Args:
        text (str): Mensaje de éxito a imprimir.
    """
    print(coloured("SUCCESS", colour='white', background='green', style='bold') + " " + coloured(text, colour='green', style='bold'))

def print_info(text):
    """
    Imprime un mensaje informativo en azul.
    
    Args:
        text (str): Mensaje informativo a imprimir.
    """
    print(coloured("INFO", colour='yellow', background='blue', style='bold') + " " + coloured(text, colour='blue', style='bold'))
    
def print_debug(text):
    """
    Imprime un mensaje de depuración en naranja.
    
    Args:
        text (str): Mensaje de depuración a imprimir.
    """
    print(coloured("DEBUG", colour='black', background='orange', style='bold') + " " + coloured(text, colour='orange', style='bold'))

def print_log(text):
    """
    Imprime un mensaje de log en cian.
    
    Args:
        text (str): Mensaje de log a imprimir.
    """
    print(coloured("LOG", colour='black', background='cyan', style='bold') + " " + coloured(text, colour='cyan', style='bold'))
    
def print_header(text):
    """
    Imprime un encabezado en negrita y azul.
    
    Args:
        text (str): Encabezado a imprimir.
    """
    print(coloured(text, colour='blue', background='yellow', style='bold'))