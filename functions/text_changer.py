
# Libraries -----------------------------------------------------------------------------------
import re
# Functions -----------------------------------------------------------------------------------

# Función para cambiar de camel_case a snake_case
def split_camel_to_snake(string, case='camel'):
    """
    Convierte una cadena en formato camelCase o PascalCase a snake_case.

    Esta función maneja la conversión de cadenas que están en formato camelCase o PascalCase
    a snake_case, donde las palabras se separan por guiones bajos ('_').

    Args:
    string (str): La cadena en formato camelCase o PascalCase que se desea convertir.
    case (str, optional): El tipo de caso de la cadena. Puede ser 'camel' (por defecto) para camelCase o 'pascal' para PascalCase.

    Returns:
    str: La cadena convertida a snake_case.

    Raises:
    None.

    Example:
    >>> convert_to_snake_case("estaEsUnaCadenaCamelCase", case='camel')
    'esta_es_una_cadena_camel_case'
    >>> convert_to_snake_case("EstaEsUnaCadenaPascalCase", case='pascal')
    'esta_es_una_cadena_pascal_case'
    """
    if case == 'camel':
        if string.islower():
            return string
        elif string.isupper():
            return '_'
        else:
            # Split camel case
            return '_'.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)).lower()
        
    elif case == 'pascal':
        if string.islower():
            return string
        else:
            # Split pascal case
            return '_'.join(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', string)).lower()