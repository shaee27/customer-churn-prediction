"""Nox sessions."""

import os
import tempfile
from typing import Any

import nox
from nox.sessions import Session


nox.options.sessions = "black", "flake8", "mypy", "tests"
locations = "src", "noxfile.py"


def install_with_constraints(
    session: Session, *args: str, **kwargs: Any
) -> None:
    """Install packages constrained by Poetry's lock file."""
    with tempfile.NamedTemporaryFile(delete=False) as requirements:
        session.run(
            "poetry",
            "export",
            "--format",
            "requirements.txt",
            "--with",
            "dev",
            "--without-hashes",
            "--output",
            f"{requirements.name}",
            external=True,
        )
        session.install(
            "--requirement", f"{requirements.name}", *args, **kwargs
        )
        requirements.close()
        os.unlink(requirements.name)


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", "--line-length", "79", *args)


@nox.session(python="3.9")
def black_check(session: Session) -> None:
    """Run black code checker without modifying files."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", "--check", "--line-length", "79", *args)


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Run flake8 code checker."""
    args = session.posargs or locations
    install_with_constraints(session, "flake8")
    session.run("flake8", *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python="3.9")
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs
    session.run("poetry", "run", "pip", "install", "json2html", external=True)
    session.run("poetry", "install", "--only", "main", external=True)
    install_with_constraints(session, "pytest")
    try:
        session.run(
            "coverage", "run", "--source", "src", "-m", "pytest", *args
        )
    finally:
        session.run("coverage", "report")
