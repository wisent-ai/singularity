#!/usr/bin/env python3
"""
Kotlin code templates for Android project generation.

Contains Gradle, Manifest, and Kotlin source templates.

Theme templates (color_kt, theme_kt, type_kt), LLM prompt builders, and
response parsers live in kotlin_helpers.py and are re-exported here.
"""

# Re-export from kotlin_helpers so existing callers continue to work
from .kotlin_helpers import (
    generation_prompt,
    add_screen_prompt,
    parse_llm_response,
    extract_kotlin_code,
    color_kt,
    theme_kt,
    type_kt,
)


def root_build_gradle() -> str:
    return '''plugins {
    id("com.android.application") version "8.2.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.20" apply false
}
'''


def settings_gradle(name: str) -> str:
    return f'''pluginManagement {{
    repositories {{
        google()
        mavenCentral()
        gradlePluginPortal()
    }}
}}
dependencyResolutionManagement {{
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {{
        google()
        mavenCentral()
    }}
}}
rootProject.name = "{name}"
include(":app")
'''


def gradle_properties() -> str:
    return '''org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
android.nonTransitiveRClass=true
'''


def app_build_gradle(package_name: str, min_sdk: int) -> str:
    return f'''plugins {{
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}}

android {{
    namespace = "{package_name}"
    compileSdk = 34

    defaultConfig {{
        applicationId = "{package_name}"
        minSdk = {min_sdk}
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {{
            useSupportLibrary = true
        }}
    }}

    buildTypes {{
        release {{
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }}
    }}
    compileOptions {{
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }}
    kotlinOptions {{
        jvmTarget = "1.8"
    }}
    buildFeatures {{
        compose = true
    }}
    composeOptions {{
        kotlinCompilerExtensionVersion = "1.5.4"
    }}
    packaging {{
        resources {{
            excludes += "/META-INF/{{AL2.0,LGPL2.1}}"
        }}
    }}
}}

dependencies {{
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    implementation("androidx.activity:activity-compose:1.8.1")
    implementation(platform("androidx.compose:compose-bom:2023.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.navigation:navigation-compose:2.7.5")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2023.10.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}}
'''


def proguard_rules() -> str:
    return '''# Add project specific ProGuard rules here.
-keepattributes SourceFile,LineNumberTable
'''


def android_manifest(name: str, display_name: str) -> str:
    return f'''<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="{display_name}"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.{name}">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:label="{display_name}"
            android:theme="@style/Theme.{name}">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
'''


def main_activity_kt(package_name: str, name: str) -> str:
    return f'''package {package_name}

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import {package_name}.ui.theme.{name}Theme

class MainActivity : ComponentActivity() {{
    override fun onCreate(savedInstanceState: Bundle?) {{
        super.onCreate(savedInstanceState)
        setContent {{
            {name}Theme {{
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {{
                    MainScreen()
                }}
            }}
        }}
    }}
}}
'''


def main_screen_kt(package_name: str, name: str, display_name: str) -> str:
    return f'''package {package_name}

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import {package_name}.ui.theme.{name}Theme

@Composable
fun MainScreen() {{
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {{
        Text(
            text = "Welcome to {display_name}",
            style = MaterialTheme.typography.headlineMedium
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Your app is ready!",
            style = MaterialTheme.typography.bodyLarge
        )
    }}
}}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {{
    {name}Theme {{
        MainScreen()
    }}
}}
'''


def strings_xml(display_name: str) -> str:
    return f'''<resources>
    <string name="app_name">{display_name}</string>
</resources>
'''


def themes_xml(name: str) -> str:
    return f'''<?xml version="1.0" encoding="utf-8"?>
<resources>
    <style name="Theme.{name}" parent="android:Theme.Material.Light.NoActionBar" />
</resources>
'''


def gradle_wrapper_properties() -> str:
    return '''distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\\://services.gradle.org/distributions/gradle-8.2-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
'''
