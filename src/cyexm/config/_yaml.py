from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.loader import SafeLoader
from yaml.nodes import MappingNode, Node, ScalarNode

from cyexm.errors import ManifestDecodeError


class _StrictSafeLoader(SafeLoader):
    def compose_node(self, parent: Node | None, index: int) -> Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ComposerError(
                None,
                None,
                "aliases are not allowed",
                event.start_mark,
            )
        return cast(Node, super().compose_node(parent, index))

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[str, object]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping node",
                node.start_mark,
            )

        mapping: dict[str, object] = {}
        for key_node, value_node in node.value:
            if (
                not isinstance(key_node, ScalarNode)
                or key_node.tag != "tag:yaml.org,2002:str"
            ):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "mapping keys must be scalar strings",
                    key_node.start_mark,
                )
            key = cast(str, self.construct_object(key_node, deep=deep))
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate key {key!r} is not allowed",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _reject_custom_tag(loader: _StrictSafeLoader, node: Node) -> object:
    raise ConstructorError(
        None,
        None,
        f"custom YAML tags are not allowed: {node.tag}",
        node.start_mark,
    )


_StrictSafeLoader.add_constructor(
    None,  # pyright: ignore[reportArgumentType] -- PyYAML uses None for unknown tags.
    _reject_custom_tag,
)


def _yaml_error_detail(error: yaml.YAMLError) -> str:
    if isinstance(error, ComposerError) and "expected a single document" in str(error):
        return "expected one YAML document"
    return str(error)


def load_yaml_mapping(path: str | Path) -> Mapping[str, object]:
    source = Path(path)
    loader: _StrictSafeLoader | None = None
    try:
        content = source.read_text(encoding="utf-8")
        loader = _StrictSafeLoader(content)
        value = loader.get_single_data()
        if not isinstance(value, Mapping):
            raise yaml.YAMLError("YAML document must contain a top-level mapping")
        return cast(Mapping[str, object], value)
    except OSError as error:
        raise ManifestDecodeError(
            f"Could not decode YAML from {source}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ManifestDecodeError(
            f"Could not decode YAML from {source}: {_yaml_error_detail(error)}"
        ) from error
    finally:
        if loader is not None:
            loader.dispose()
