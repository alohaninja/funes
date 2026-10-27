//! Explicit CLI status reads the named memory, independent of the host binding and update check.

mod support;

use funes::inference::EmbeddingModel;
use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output, Stdio};

fn status(home: &Path, funes_home: &Path, proxy: &TcpListener, args: &[&str]) -> Output {
    let proxy_url = format!("http://{}", proxy.local_addr().unwrap());
    Command::new(env!("CARGO_BIN_EXE_funes"))
        .current_dir(home)
        .arg("status")
        .args(args)
        .env("HOME", home)
        .env("FUNES_HOME", funes_home)
        .env("HF_HOME", home.join("hf"))
        .env("DO_NOT_TRACK", "1")
        .env("HTTP_PROXY", &proxy_url)
        .env("HTTPS_PROXY", &proxy_url)
        .env("ALL_PROXY", &proxy_url)
        .env("http_proxy", &proxy_url)
        .env("https_proxy", &proxy_url)
        .env("all_proxy", &proxy_url)
        .env("NO_PROXY", "")
        .env("no_proxy", "")
        .env_remove("HF_TOKEN")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .env_remove("HUGGINGFACE_TOKEN")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn proxy() -> TcpListener {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    listener
}

#[tokio::test]
async fn local_status_reads_the_index_without_contacting_the_bound_remote_or_update_server() {
    let home = tempfile::tempdir().unwrap();
    let funes_home = home.path().join("funes");
    fs::create_dir_all(&funes_home).unwrap();
    support::memory_of(
        EmbeddingModel::BgeSmallEn,
        &["a local turn awaiting the next publication"],
        &funes_home.join("memory"),
    )
    .await;
    fs::write(funes_home.join("bound"), "acme/status-test").unwrap();
    let network = proxy();
    let out = status(home.path(), &funes_home, &network, &["--memory", "local"]);
    support::assert_success(&out);
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.starts_with(&format!("memory: {}\nchunks: 1\n", funes_home.join("memory").display())),
        "{text}"
    );
    assert!(text.contains("sessions: 1\n"), "{text}");
    assert!(
        !text.contains("acme/status-test") && !text.contains("local push:"),
        "{text}"
    );
    assert!(network.accept().is_err(), "scoped local status made a network request");

    // Verify the proxy really observes the ordinary update check; absence above must not be a
    // disconnected spy. No bound remote here, so the only network work is the update check.
    fs::remove_file(funes_home.join("bound")).unwrap();
    let control = status(home.path(), &funes_home, &network, &[]);
    support::assert_success(&control);
    assert!(
        network.accept().is_ok(),
        "the proxy did not observe the default status update check"
    );
}

#[test]
fn explicitly_named_status_does_not_require_a_readable_binding_and_keeps_stale_install_notes() {
    let home = tempfile::tempdir().unwrap();
    let funes_home = home.path().join("funes");
    fs::create_dir_all(funes_home.join("bound")).unwrap();
    fs::create_dir_all(funes_home.join("spool")).unwrap();
    fs::write(funes_home.join("spool/bonnie.missing"), "").unwrap();
    let network = proxy();
    let out = status(home.path(), &funes_home, &network, &["--memory", "local"]);
    support::assert_success(&out);
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("no index yet"), "{text}");
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(
        err.contains("the bonnie integration does not match this version of funes"),
        "{err}"
    );
    assert!(!text.contains("note:"), "{text}");
    assert!(network.accept().is_err(), "scoped local status made a network request");
}

#[tokio::test]
async fn an_explicit_path_reports_that_memory_instead_of_the_host_index() {
    let home = tempfile::tempdir().unwrap();
    let funes_home = home.path().join("funes");
    fs::create_dir_all(&funes_home).unwrap();
    let other = home.path().join("other");
    support::memory_of(EmbeddingModel::BgeSmallEn, &["the other memory"], &other).await;
    support::memory_of(
        EmbeddingModel::BgeSmallEn,
        &["the host's memory"],
        &funes_home.join("memory"),
    )
    .await;
    support::write_session(home.path(), "pending", &["a host turn still waiting to be indexed"]);
    fs::write(
        funes_home.join("index-coverage.json"),
        serde_json::json!({"pending": [home.path().join("pending.funes.jsonl")]}).to_string(),
    )
    .unwrap();
    let network = proxy();
    let out = status(
        home.path(),
        &funes_home,
        &network,
        &["--memory", other.to_str().unwrap()],
    );
    support::assert_success(&out);
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.starts_with(&format!("memory: {}\nchunks: 1\n", other.display())),
        "{text}"
    );
    assert!(!text.contains("no index yet"), "{text}");
    assert!(
        !text.contains("pending indexing:"),
        "another memory reported the host's backlog: {text}"
    );
    assert!(network.accept().is_err(), "scoped path status made a network request");

    let local = status(home.path(), &funes_home, &network, &["--memory", "local"]);
    support::assert_success(&local);
    let text = String::from_utf8(local.stdout).unwrap();
    assert!(
        text.contains("pending indexing: 1 source session"),
        "the host still reports its backlog: {text}"
    );
    let alias = status(home.path(), &funes_home, &network, &["--memory", "./funes/memory"]);
    support::assert_success(&alias);
    let text = String::from_utf8(alias.stdout).unwrap();
    assert!(
        text.contains("pending indexing: 1 source session"),
        "a relative path to the host index lost its backlog: {text}"
    );
}
