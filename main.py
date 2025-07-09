#!/usr/bin/env python3
"""
AST Parser for expressions with assignment support and REPL with history.

Grammar:
    statement     ::= assignment | expression
    assignment    ::= NAME '=' expression
    expression    ::= primary_expr ('[' arg_list ']')*
    primary_expr  ::= NAME | NAME '[' arg_list ']'
    arg_list      ::= expression (',' expression)*
    NAME          ::= [a-zA-Z_][a-zA-Z0-9_]*
"""

import sys
from dataclasses import dataclass, field
from typing import List, Union, Optional, Mapping, MutableMapping
from enum import Enum, auto


# AST Node Types
class NodeType(Enum):
    IDENTIFIER = auto()
    EXPRESSION = auto()
    ASSIGNMENT = auto()
    CHAINED_EXPRESSION = auto()


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    type: NodeType


@dataclass
class Identifier(ASTNode):
    """Represents a simple identifier/argument."""
    name: str
    type: NodeType = field(init=False)

    def __post_init__(self):
        self.type = NodeType.IDENTIFIER

    def __str__(self):
        return self.name


@dataclass
class Expression(ASTNode):
    """Represents an expression with optional arguments."""
    name: str
    type: NodeType = field(init=False)
    args: List['ASTNode']

    def __post_init__(self):
        self.type = NodeType.EXPRESSION

    def __str__(self):
        if not self.args:
            return self.name
        args_str = ', '.join(str(arg) for arg in self.args)
        return f"{self.name}[{args_str}]"


@dataclass
class ChainedExpression(ASTNode):
    """Represents a chained expression like Bind[x, y, Expr[x, y]][a, b]."""
    base: ASTNode
    type: NodeType = field(init=False)
    arg_lists: List[List['ASTNode']]

    def __post_init__(self):
        self.type = NodeType.CHAINED_EXPRESSION

    def __str__(self):
        result = str(self.base)
        for args in self.arg_lists:
            args_str = ', '.join(str(arg) for arg in args)
            result += f"[{args_str}]"
        return result


@dataclass
class Assignment(ASTNode):
    """Represents an assignment statement."""
    name: str
    type: NodeType = field(init=False)
    value: ASTNode

    def __post_init__(self):
        self.type = NodeType.ASSIGNMENT

    def __str__(self):
        return f"{self.name} = {self.value}"


# Token Types
class TokenType(Enum):
    NAME = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    EQUALS = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int


class Lexer:
    """Tokenizer for the expression language."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens = []
        self._tokenize()

    def _tokenize(self):
        """Convert input text into tokens."""
        text = self.text.strip()
        i = 0

        while i < len(text):
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue

            # Single character tokens
            if text[i] == '[':
                self.tokens.append(Token(TokenType.LBRACKET, '[', i))
                i += 1
            elif text[i] == ']':
                self.tokens.append(Token(TokenType.RBRACKET, ']', i))
                i += 1
            elif text[i] == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', i))
                i += 1
            elif text[i] == '=':
                self.tokens.append(Token(TokenType.EQUALS, '=', i))
                i += 1
            elif text[i].isalpha() or text[i] == '_':
                # Identifier/Name
                start = i
                while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                    i += 1
                name = text[start:i]
                self.tokens.append(Token(TokenType.NAME, name, start))
            else:
                raise SyntaxError(f"Unexpected character '{text[i]}' at position {i}")

        self.tokens.append(Token(TokenType.EOF, '', len(text)))


class Parser:
    """Recursive descent parser for the expression language."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        """Get the current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF token

    def consume(self, expected_type: TokenType) -> Token:
        """Consume a token of the expected type."""
        token = self.current_token()
        if token.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token.type} at position {token.pos}")
        self.pos += 1
        return token

    def peek(self) -> TokenType:
        """Peek at the current token type."""
        return self.current_token().type

    def parse(self) -> ASTNode:
        """Parse the input into an AST."""
        if self.peek() == TokenType.EOF:
            raise SyntaxError("Empty input")

        # Check if this is an assignment
        if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.EQUALS:
            return self.parse_assignment()
        else:
            return self.parse_expression()

    def parse_assignment(self) -> Assignment:
        """Parse an assignment statement: NAME = expression."""
        name_token = self.consume(TokenType.NAME)
        self.consume(TokenType.EQUALS)
        value = self.parse_expression()
        return Assignment(name_token.value, value)

    def parse_expression(self) -> ASTNode:
        """Parse an expression with potential chaining: primary_expr ('[' arg_list ']')*."""
        base = self.parse_primary_expression()

        # Check for chaining (additional argument lists)
        arg_lists = []
        while self.peek() == TokenType.LBRACKET:
            self.consume(TokenType.LBRACKET)
            args = self.parse_arg_list()
            self.consume(TokenType.RBRACKET)
            arg_lists.append(args)

        if arg_lists:
            return ChainedExpression(base, arg_lists)
        else:
            return base

    def parse_primary_expression(self) -> Union[Expression, Identifier]:
        """Parse a primary expression: NAME or NAME[arg_list]."""
        name_token = self.consume(TokenType.NAME)

        if self.peek() == TokenType.LBRACKET:
            # Expression with arguments
            self.consume(TokenType.LBRACKET)
            args = self.parse_arg_list()
            self.consume(TokenType.RBRACKET)
            return Expression(name_token.value, args)
        else:
            # Simple identifier
            return Identifier(name_token.value)

    def parse_arg_list(self) -> List[ASTNode]:
        """Parse a comma-separated list of arguments."""
        args = []

        if self.peek() == TokenType.RBRACKET:
            # Empty argument list
            return args

        args.append(self.parse_expression())

        while self.peek() == TokenType.COMMA:
            self.consume(TokenType.COMMA)
            args.append(self.parse_expression())

        return args


TRUE = Identifier("True")
FALSE = Identifier("False")


class Evaluator:
    def __init__(self, env: Optional[Mapping[str, ASTNode]] = None):
        self.environment: MutableMapping[str, ASTNode] = env or {}

    def reduce_inner(self, expr: ASTNode) -> ASTNode:
        new_expr = self.evaluate_inner(expr)
        if isinstance(new_expr, Expression):
            if new_expr.name == "And":
                current_expr = self.reduce_and(new_expr)
                return current_expr
            elif new_expr.name == "Or":
                current_expr = self.reduce_or(new_expr)
                return current_expr
            elif new_expr.name == "Equal":
                current_expr = self.reduce_equal(new_expr)
                return current_expr
            elif new_expr.name == "Not":
                current_expr = self.reduce_not(new_expr)
                return current_expr
            else:
                return new_expr
        elif isinstance(new_expr, Identifier):
            return new_expr
        else:
            return new_expr

    def reduce_equal(self, expr: ASTNode) -> ASTNode:
        assert isinstance(expr, Expression)
        assert len(expr.args) == 2
        new_a = self.reduce_inner(expr.args[0])
        new_b = self.reduce_inner(expr.args[1])
        if new_a == new_b:
            return TRUE
        else:
            return FALSE

    def reduce_not(self, expr: ASTNode) -> ASTNode:
        assert isinstance(expr, Expression)
        assert len(expr.args) == 1
        new_arg = self.reduce_inner(expr.args[0])
        if new_arg == TRUE:
            return FALSE
        elif new_arg == FALSE:
            return TRUE
        else:
            return expr

    def reduce_or(self, expr: ASTNode) -> ASTNode:
        assert isinstance(expr, Expression)
        assert len(expr.args) == 2
        a, b = expr.args[0], expr.args[1]
        new_a = self.reduce_inner(a)
        new_b = self.reduce_inner(b)
        if new_a == TRUE or new_b == TRUE:
            return TRUE
        elif new_a == FALSE and new_b == FALSE:
            return FALSE
        else:
            return Expression(name="Or", args=[new_a, new_b])

    def reduce_and(self, expr: ASTNode) -> ASTNode:
        assert isinstance(expr, Expression)
        assert len(expr.args) == 2
        a, b = expr.args[0], expr.args[1]
        new_a = self.reduce_inner(a)
        new_b = self.reduce_inner(b)
        if new_a == TRUE and new_b == TRUE:
            return TRUE
        elif new_a == FALSE or new_b == FALSE:
            return FALSE
        else:
            return Expression(name="And", args=[new_a, new_b])

    def evaluate(self, expr: ASTNode) -> ASTNode:
        if isinstance(expr, Assignment):
            self.environment[expr.name] = expr.value
            return expr

        new_expr = self._update_with_map(expr, self.environment)

        if isinstance(new_expr, Expression) and new_expr.name == "Evaluate" and len(new_expr.args) == 1:
            return self.evaluate_inner(new_expr.args[0])
        elif isinstance(new_expr, Expression) and new_expr.name == "Reduce" and len(new_expr.args) == 1:
            return self.reduce_inner(new_expr.args[0])
        else:
            return new_expr

    def evaluate_inner(self, expr: ASTNode, var_map: Optional[Mapping[str, ASTNode]] = None) -> ASTNode:
        arg = expr
        if isinstance(arg, ChainedExpression):
            """Evaluate[ Bind[x, y, Expr[x, y]][a, b] ]"""
            if isinstance(arg.base, Expression) and arg.base.name == "Bind":
                current_expr, new_var_map = self.evaluate_bind(arg, var_map)
                if current_expr != arg:
                    return self.evaluate_inner(current_expr, new_var_map)
                else:
                    return current_expr
            elif isinstance(arg.base, Expression) and arg.base.name == "Recurse":
                current_expr = self.evaluate_recurse(arg, var_map)
                while isinstance(current_expr, (Expression, ChainedExpression)):
                    if (ret_expr := self.evaluate_inner(current_expr, var_map)) != current_expr:
                        current_expr = ret_expr
                    else:
                        if (ret_expr := self._update_with_map(current_expr, self.environment)) != current_expr:
                            current_expr = ret_expr
                        else:
                            break
                return current_expr
            else:
                return arg
        elif isinstance(arg, Expression):
            """Add[Zero, Zero]"""
            new_args = [self.evaluate_inner(a) for a in arg.args]
            return Expression(name=arg.name, args=new_args)
        else:
            return arg

    def evaluate_bind(self, expr: ASTNode, var_map: Optional[Mapping[str, ASTNode]] = None) -> (ASTNode, Mapping[str, ASTNode]):
        """Evaluate[ Bind[x, y, Expr[x, y]][a, b] ]"""
        assert isinstance(expr, ChainedExpression)
        assert isinstance(expr.base, Expression) and expr.base.name == "Bind"
        assert len(expr.arg_lists) == 1
        assert len(expr.base.args) == len(expr.arg_lists[0]) + 1
        variables = expr.base.args[:-1]
        var_names = []
        for var in variables:
            assert isinstance(var, Identifier)
            var_names.append(var.name)

        new_var_map = var_map or {}
        new_var_map.update(dict((name, val) for name, val in zip(var_names, expr.arg_lists[0])))

        body_expr = expr.base.args[-1]
        return self._update_with_map(body_expr, new_var_map), new_var_map

    def evaluate_recurse(self, expr: ASTNode, var_map: Optional[Mapping[str, ASTNode]] = None) -> ASTNode:
        assert isinstance(expr, ChainedExpression)
        assert isinstance(expr.base, Expression) and expr.base.name == "Recurse"
        assert len(expr.arg_lists) == 1
        arg = expr.arg_lists[0][0]

        if not (isinstance(arg, Expression) and arg.name == "Succ"
                or isinstance(arg, Identifier) and arg.name == "Zero"):
            arg = self.evaluate_inner(arg, var_map)

        pm = PeanoMatcher()
        assert pm.accepts(arg)
        assert isinstance(expr.base.args[0], Identifier)
        variable = expr.base.args[0]
        assert isinstance(variable, Identifier)
        var_name = variable.name

        recursive_case = expr.base.args[1]
        base_case = expr.base.args[2]

        if isinstance(arg, Identifier) and arg.name == "Zero":
            return base_case

        if isinstance(recursive_case, Identifier):
            if recursive_case.name == var_name:
                return arg
            else:
                return recursive_case

        assert isinstance(recursive_case, Expression)

        recursor = base_case
        current_arg = arg
        while not isinstance(current_arg, Identifier):
            assert isinstance(current_arg, Expression)
            new_var_map = var_map or {}
            new_var_map.update({"Self": recursor, var_name: current_arg})
            recursor = self._update_with_map(recursive_case, new_var_map)
            current_arg = current_arg.args[0]

        return recursor

    def _update_with_map(self, expr: ASTNode, var_map: Mapping[str, ASTNode]) -> ASTNode:
        if isinstance(expr, Identifier):
            if expr.name in var_map:
                return var_map[expr.name]
            else:
                return expr
        elif isinstance(expr, Expression):
            if expr.name in var_map:
                new_node = var_map[expr.name]
                if isinstance(new_node, Identifier):
                    # x[y, z] + A -> A[y, z]
                    args = []
                    for node in expr.args:
                        args.append(self._update_with_map(node, var_map))
                    return Expression(name=new_node.name, args=args)
                elif isinstance(new_node, Expression):
                    # x[y, z] + A[B] -> A[B][y, z]
                    args = []
                    for node in expr.args:
                        args.append(self._update_with_map(node, var_map))
                    return ChainedExpression(base=new_node, arg_lists=[args])
                else:
                    raise SyntaxError
            else:
                # Replace args only
                args = []
                for node in expr.args:
                    args.append(self._update_with_map(node, var_map))
                return Expression(name=expr.name, args=args)
        elif isinstance(expr, ChainedExpression):
            assert isinstance(expr.base, Expression)
            assert expr.base.name in ("Recurse", "Bind")
            assert len(expr.arg_lists) == 1
            if expr.base.name == "Recurse":
                base_case = self._update_with_map(expr.base.args[2], var_map)
                new_base = Expression(name="Recurse", args=expr.base.args[:2] + [base_case])
            else:
                new_base = expr.base
            args = []
            for node in expr.arg_lists[0]:
                args.append(self._update_with_map(node, var_map))
            return ChainedExpression(base=new_base, arg_lists=[args])
        return expr


class ExpressionParser:
    """Main parser class that combines lexing and parsing."""

    def parse(self, text: str) -> ASTNode:
        """Parse a text string into an AST."""
        if not text.strip():
            raise SyntaxError("Empty input")

        try:
            lexer = Lexer(text)
            parser = Parser(lexer.tokens)
            return parser.parse()
        except Exception as e:
            raise SyntaxError(f"Parse error: {e}")


class ExpressionMatcher:
    pass


class PeanoMatcher(ExpressionMatcher):
    def __init__(self):
        self.state = "Start"
        self.transitions = {"Start": [], "Read": [], "Accept": []}
        self.transitions["Start"].append((Expression("Succ", args=[Identifier("Any")]), "Read"))
        self.transitions["Read"].append((Expression("Succ", args=[Identifier("Any")]), "Read"))
        self.transitions["Start"].append((Identifier("Zero"), "Accept"))
        self.transitions["Read"].append((Identifier("Zero"), "Accept"))

    def accepts(self, expr: ASTNode) -> bool:
        visited = []
        current_expr = expr
        while current_expr not in visited:
            visited.append(current_expr)
            for match_expr, new_state in self.transitions[self.state]:
                if (new_expr := self.match(match_expr, current_expr)) is not None:
                    self.state = new_state
                    current_expr = new_expr
                    if new_state == "Accept":
                        return True

                    break

        return False

    def match(self, match_expr: ASTNode, expr: ASTNode) -> Optional[ASTNode]:
        if isinstance(match_expr, Expression) and isinstance(expr, Expression):
            if match_expr.name != expr.name:
                return None

            for marg, arg in zip(match_expr.args, expr.args):
                if (new_expr := self.match(marg, arg)) is not None:
                    return new_expr

            return None

        elif isinstance(match_expr, Identifier):
            if match_expr.name == "Any":
                return expr
            elif isinstance(expr, Identifier) and expr.name == match_expr.name:
                return expr
            else:
                return None

        return None


# REPL with History Support
class REPL:
    """Read-Eval-Print Loop with history support."""

    def __init__(self):
        self.parser = ExpressionParser()
        self.evaluator = Evaluator()
        self.history = []
        self.history_pos = 0
        self.setup_readline()

    def setup_readline(self):
        """Setup readline for history and arrow key support."""
        try:
            import readline
            import atexit

            # History file
            history_file = ".expr_history"

            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass

            # Save history on exit
            atexit.register(readline.write_history_file, history_file)

            # Enable history search
            readline.parse_and_bind('tab: complete')

            self.readline_available = True
        except ImportError:
            print("Warning: readline not available. History and arrow keys disabled.")
            self.readline_available = False

    def get_input(self, prompt: str) -> str:
        """Get input with history support."""
        if self.readline_available:
            import readline
            return input(prompt)
        else:
            return input(prompt)

    def run(self):
        """Run the REPL."""
        print("Expression Parser REPL")
        print("Supported syntax:")
        print("  - Simple args: Arg1")
        print("  - Expressions: Expr, Expr[A], Expr[A, B]")
        print("  - Nested: Expr[NestedExpr[Arg]]")
        print("  - Assignment: Name = Expr[A]")
        print("  - Commands: :help, :history, :quit")
        print()

        while True:
            try:
                line = self.get_input(">>> ").strip()

                if not line:
                    continue

                # Handle special commands
                if line.startswith(':'):
                    if line == ':quit' or line == ':q':
                        print("Goodbye!")
                        break
                    elif line == ':help' or line == ':h':
                        self.show_help()
                        continue
                    elif line == ':history':
                        self.show_history()
                        continue
                    else:
                        print(f"Unknown command: {line}")
                        continue

                # Parse input and display result
                ast = self.parser.parse(line)
                result = self.evaluator.evaluate(ast)
                print(result)

                # Add to history
                self.history.append(line)

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except SyntaxError as e:
                print(f"Syntax Error: {e}")
            except Exception as e:
                print(f"Error: {e}")

    def show_help(self):
        """Show help information."""
        print("Available commands:")
        print("  :help, :h     - Show this help")
        print("  :history      - Show command history")
        print("  :quit, :q     - Exit the REPL")
        print()
        print("Expression syntax:")
        print("  Arg1                    - Simple identifier")
        print("  Expr                    - Expression without arguments")
        print("  Expr[A]                 - Expression with one argument")
        print("  Expr[A, B]              - Expression with multiple arguments")
        print("  Expr[NestedExpr[Arg]]   - Nested expressions")
        print("  Name = Expr[A]          - Assignment")

    def show_history(self):
        """Show command history."""
        if not self.history:
            print("No history available.")
            return

        print("Command history:")
        for i, cmd in enumerate(self.history, 1):
            print(f"  {i:2d}: {cmd}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Parse command line argument
        text = ' '.join(sys.argv[1:])
        parser = ExpressionParser()
        try:
            ast = parser.parse(text)
            print(f"Input: {text}")
            print(f"AST: {ast}")
            print(f"Type: {ast.type.name}")
        except SyntaxError as e:
            print(f"Syntax Error: {e}")
            sys.exit(1)
    else:
        # Start REPL
        repl = REPL()
        repl.run()


if __name__ == "__main__":
    main()
