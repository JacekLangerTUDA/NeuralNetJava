plugins {
    id("java")
//    kotlin("jvm") version "1.6.21"
//    kotlin("plugin.spring") version "1.6.21"
    // https://github.com/diffplug/spotless
    // https://plugins.gradle.org/plugin/com.diffplug.gradle.spotless
    id("com.diffplug.spotless") version ("6.11.0")
}

group = "org.example"
// java.sourceCompatibility = JavaVersion.VERSION_17
version = "0.1.0-SNAPSHOT"

configurations {
    compileOnly {
        extendsFrom(configurations.annotationProcessor.get())
    }
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains:annotations:20.1.0")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.1")

    implementation("org.apache.logging.log4j:log4j-api:2.19.0")
    implementation("org.apache.logging.log4j:log4j-core:2.19.0")

    // https://mvnrepository.com/artifact/org.slf4j/slf4j-api
//    implementation("org.slf4j:slf4j-api:2.0.3")

    // https://mvnrepository.com/artifact/com.google.code.gson/gson
    implementation("com.google.code.gson:gson:2.10")

    implementation("me.tongfei:progressbar:0.9.5")
}

// tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
//    kotlinOptions {
//        freeCompilerArgs = listOf("-Xjsr305=strict")
//        jvmTarget = "17"
//    }
// }

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}
