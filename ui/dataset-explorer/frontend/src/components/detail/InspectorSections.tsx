import PythonCodeBlock from "@/components/code/PythonCodeBlock";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/code-block";
import type { DerivedFieldSummary, InspectorSection } from "@/types/datasetExplorer";

// Add new derived fields here when they should render with code syntax highlighting.
export const CODE_DERIVED_FIELD_NAMES = new Set(["Task.gt_solution"]);

export function isCodeDerivedField(name: string) {
  return CODE_DERIVED_FIELD_NAMES.has(name);
}

function renderValue(sectionKey: string, kind: string, value: unknown) {
  if (value === null || value === undefined) {
    return <p className="text-sm text-muted-foreground">None</p>;
  }

  if (typeof value === "string") {
    if (kind === "code") {
      return <PythonCodeBlock code={value} />;
    }

    if (kind === "error") {
      return <CodeBlock variant="error">{value}</CodeBlock>;
    }

    return <CodeBlock>{value}</CodeBlock>;
  }

  return <CodeBlock data-testid={`json-${sectionKey}`}>{JSON.stringify(value, null, 2)}</CodeBlock>;
}

export function DerivedFieldsCard({ derivedFields }: { derivedFields: DerivedFieldSummary[] }) {
  if (!derivedFields.length) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Derived Field Mapping</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {derivedFields.map((field) => (
          <div key={field.name} className="rounded-md border p-3">
            <div className="text-sm font-medium">{field.name}</div>
            <div className="mt-1 text-xs text-muted-foreground">{field.source}</div>
            <div className="mt-3">
              {isCodeDerivedField(field.name) ? (
                <PythonCodeBlock code={field.value} className="p-0" />
              ) : (
                <CodeBlock className="p-3">{field.value}</CodeBlock>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function InspectorSections({ sections }: { sections: InspectorSection[] }) {
  return (
    <div className="space-y-6">
      {sections.map((section) => (
        <Card key={section.title}>
          <CardHeader>
            <CardTitle className="text-base">{section.title}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            {section.fields.map((field) => (
              <div key={`${section.title}-${field.key}`} className="space-y-2">
                <div className="text-sm font-medium">{field.label}</div>
                {renderValue(field.key, field.kind, field.value)}
              </div>
            ))}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
