#!/usr/bin/env python3
"""
Swift code templates for iOS app generation.

Contains SwiftUI view templates, Info.plist, LLM prompt builders,
and LLM response parsing utilities.
The large pbxproj_template lives in swift_helpers.py.
"""

import re
from typing import Dict, List, Optional

# Re-export pbxproj_template so existing callers continue to work
from .swift_helpers import pbxproj_template


def app_file_template(name: str) -> str:
    """Generate the main App entry point Swift file."""
    return f'''import SwiftUI

@main
struct {name}App: App {{
    var body: some Scene {{
        WindowGroup {{
            ContentView()
        }}
    }}
}}
'''


def content_view_template(display_name: str) -> str:
    """Generate the default ContentView Swift file."""
    return f'''import SwiftUI

struct ContentView: View {{
    var body: some View {{
        NavigationStack {{
            VStack(spacing: 20) {{
                Image(systemName: "star.fill")
                    .imageScale(.large)
                    .foregroundStyle(.tint)

                Text("Welcome to {display_name}")
                    .font(.title)
            }}
            .padding()
            .navigationTitle("{display_name}")
        }}
    }}
}}

#Preview {{
    ContentView()
}}
'''


def content_view_redirect_template(main_view_name: str) -> str:
    """Generate a ContentView that redirects to a named main view."""
    return f'''import SwiftUI

struct ContentView: View {{
    var body: some View {{
        {main_view_name}()
    }}
}}

#Preview {{
    ContentView()
}}
'''


def info_plist_template(display_name: str, bundle_id: str) -> str:
    """Generate the Info.plist XML content."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>$(DEVELOPMENT_LANGUAGE)</string>
    <key>CFBundleDisplayName</key>
    <string>{display_name}</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>{bundle_id}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>$(PRODUCT_BUNDLE_PACKAGE_TYPE)</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSRequiresIPhoneOS</key>
    <true/>
    <key>UIApplicationSceneManifest</key>
    <dict>
        <key>UIApplicationSupportsMultipleScenes</key>
        <true/>
    </dict>
    <key>UILaunchScreen</key>
    <dict/>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>
</dict>
</plist>
'''


def add_view_prompt(view_name: str, description: str) -> str:
    """Build the LLM prompt for adding a single SwiftUI view."""
    return f"""Generate a SwiftUI view named {view_name} with the following functionality:

{description}

Requirements:
- Use modern SwiftUI (iOS 16+)
- Follow Apple Human Interface Guidelines
- Include appropriate state management (@State, @Binding as needed)
- Add helpful comments
- Include a #Preview at the bottom

Return ONLY the Swift code, no explanations. Start with 'import SwiftUI'."""


def generation_prompt(
    description: str,
    features_text: str,
    theme_text: str,
) -> str:
    """Build the LLM prompt for full app generation."""
    return f"""Generate a complete iOS app using SwiftUI based on this description:

{description}

Features to include:
{features_text}
{theme_text}
Requirements:
1. Create multiple SwiftUI views as needed
2. Use modern SwiftUI (iOS 16+)
3. Follow Apple Human Interface Guidelines
4. Include proper navigation (NavigationStack)
5. Use appropriate state management
6. Add helpful comments
7. Each view should have a #Preview

Format your response as multiple Swift files, each starting with:
// FILE: ViewName.swift
import SwiftUI
...

Generate 2-4 views depending on complexity. The first view will be the main view."""


def build_generation_prompt(
    description: str,
    features: List[str],
    theme: Optional[Dict],
) -> str:
    """Build the full prompt for app generation from raw inputs."""
    features_text = "\n".join(f"- {f}" for f in features) if features else "None specified"
    theme_text = ""
    if theme:
        theme_text = (
            f"\nTheme:\n- Primary color: {theme.get('primary', 'blue')}\n"
            f"- Secondary color: {theme.get('secondary', 'gray')}\n"
            f"- Background: {theme.get('background', 'white')}\n"
        )
    return generation_prompt(description, features_text, theme_text)


def parse_llm_response(response: str) -> List[Dict]:
    """Parse LLM response to extract Swift view definitions."""
    views = []
    file_pattern = r'// FILE:\s*(\w+)\.swift\s*\n(.*?)(?=// FILE:|$)'
    matches = re.findall(file_pattern, response, re.DOTALL)
    for name, code in matches:
        code = code.strip()
        if code and 'import SwiftUI' in code:
            views.append({'name': name, 'code': code})
    # If no file markers found, try to extract as single view
    if not views and 'import SwiftUI' in response:
        struct_match = re.search(r'struct\s+(\w+)\s*:', response)
        if struct_match:
            views.append({'name': struct_match.group(1), 'code': extract_swift_code(response)})
    return views


def extract_swift_code(response: str) -> str:
    """Extract Swift code from LLM response."""
    code_match = re.search(r'```swift\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    if 'import SwiftUI' in response:
        start = response.find('import SwiftUI')
        return response[start:].strip()
    return response.strip()
