# advanced_example.py
from hippycampus.spec_parser import parse_yaml_file


def analyze_schema_inheritance():
    """Analyze schema inheritance and composition in an OpenAPI spec."""
    spec = parse_yaml_file('../test/ecommerce_api.yaml')

    # Get components schemas
    schemas = spec.components.schemas

    # Find schemas using inheritance (allOf)
    print("Schemas using inheritance (allOf):")
    for name, schema in schemas.items():
        if hasattr(schema, 'allOf') and schema.allOf:
            print(f"\n{name}:")

            # Show what this schema inherits from
            for i, parent_schema in enumerate(schema.allOf):
                if hasattr(parent_schema, 'ref'):
                    # This would be a reference, but it's been resolved automatically
                    parent_name = parent_schema.ref.split('/')[-1]
                    print(f"  Inherits from: {parent_name}")
                else:
                    print(f"  Has inline schema #{i + 1}")

                # Show properties from this part of the inheritance chain
                props = getattr(parent_schema, 'properties', None)
                if props:
                    print("    Properties:")
                    for prop_name, prop in props.items():
                        required = prop_name in (getattr(parent_schema, 'required', []) or [])
                        print(f"      - {prop_name}{' (required)' if required else ''}")

    # Find schemas using composition (oneOf, anyOf)
    print("\nSchemas using composition (oneOf/anyOf):")
    for name, schema in schemas.items():
        if hasattr(schema, 'oneOf') and schema.oneOf:
            print(f"\n{name} (oneOf):")
            for i, option in enumerate(schema.oneOf):
                print(f"  Option {i + 1}: {getattr(option, 'title', 'Unnamed schema')}")

        if hasattr(schema, 'anyOf') and schema.anyOf:
            print(f"\n{name} (anyOf):")
            for i, option in enumerate(schema.anyOf):
                print(f"  Option {i + 1}: {getattr(option, 'title', 'Unnamed schema')}")


def analyze_operations():
    """Analyze operations and their responses in an OpenAPI spec."""
    spec = parse_yaml_file('../test/ecommerce_api.yaml')

    # Collect all operations across all paths
    operations = []
    for path, path_item in spec.paths:
        for method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace']:
            operation = getattr(path_item, method, None)
            if operation:
                operations.append((path, method, operation))

    # Analyze operations
    print(f"Found {len(operations)} operations across {len(spec.paths.root)} paths\n")

    for path, method, operation in operations:
        print(f"{method.upper()} {path}:")
        print(f"  Summary: {operation.summary or 'No summary'}")
        print(f"  Operation ID: {operation.operationId or 'No operationId'}")

        # Show request body if present
        if operation.requestBody:
            print("  Request Body:")
            for content_type, media_type in operation.requestBody.content.items():
                print(f"    Media Type: {content_type}")
                if media_type.schema_:
                    schema_name = "Inline schema"
                    if hasattr(media_type.schema_, 'ref'):
                        schema_name = media_type.schema_.ref.split('/')[-1]
                    print(f"    Schema: {schema_name}")

        # Show responses
        print("  Responses:")
        for status_code, response in operation.responses:
            print(f"    {status_code}: {response.description}")

            # Show response content if present
            if response.content:
                for content_type, media_type in response.content.items():
                    print(f"      Media Type: {content_type}")
                    if media_type.schema_:
                        schema_name = "Inline schema"
                        if hasattr(media_type.schema_, 'ref'):
                            schema_name = media_type.schema_.ref.split('/')[-1]
                        print(f"      Schema: {schema_name}")

        print()


if __name__ == "__main__":
    analyze_operations()
    analyze_schema_inheritance()
