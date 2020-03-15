use log::*;
use serde_derive::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::{EnumIter, ToString};
use yew::format::{Binary, Nothing};
use yew::services::storage::{Area, StorageService};
use yew::services::fetch::{FetchService, FetchTask, Request, Response};
use yew::prelude::*;
use anyhow::Error;

//const KEY: &str = "yew.yahtzee.self";

pub struct App {
    fetch_service: FetchService,
    link: ComponentLink<App>,
    data: Option<Vec<u8>>,
    ft: Option<FetchTask>,
}
pub enum Msg {
    FetchData,
    FetchResourceComplete(Result<Vec<u8>, Error>),
    FetchResourceFailed,
}

impl App {
    pub fn fetch_data(&mut self) -> yew::services::fetch::FetchTask {

        let callback = self.link.callback(
            move |response: Response<Binary>| {
                let (meta, data) = response.into_parts();
                info!("META: {:?}, {:?}", meta, data);
                if meta.status.is_success() {
                    Msg::FetchResourceComplete(data)
                } else {
                    info!("failed");
                    info!("{:?}", meta);
                    info!("{:?}", data);
                    match data {
                        Ok(data) => {
                            info!("ok: {:?}", data);
                        },
                        Err(err) => {
                            info!("err: {:?}", err);
                        }
                    }
                    Msg::FetchResourceFailed // FIXME: Handle this error accordingly.
                }
            },
        );
        let request = Request::get("http://localhost:8080/scores.yht").body(Nothing).unwrap();
        let res = self.fetch_service.fetch_binary(request, callback);
        info!("{:?}", res);
        res.unwrap()
    }
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_props: Self::Properties, link: ComponentLink<Self>) -> Self {
                        
        App {
            fetch_service: FetchService::new(),
            link,
            data: None,
            ft: None,
        }
    }

    fn view(&self) -> Html {
        html! {
            <div class="yahtzee-wrapper">
                <section class="yahtzeeapp">
                    <header class="header">
                        <h1>{ "Yahtzee Advice" }</h1>
                    </header>
                    <section class="main">
                    <button onclick=self.link.callback(|_| Msg::FetchData)>
                    { "Fetch Data" }
                </button>
                    </section>
                    <footer class="footer">
                    </footer>
                </section>
                <footer class="info">
                    <p>{ "See writeup: " }<a href="https://github.com/jonstites/" target="_blank">{ "Jonathan Stites" }</a></p>
                </footer>
            </div>
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            Msg::FetchResourceComplete(response) => {
                println!("{:?}", response);
                true
            },
            Msg::FetchResourceFailed => {
                panic!("uh ohhhh");
            },
            Msg::FetchData => {
                self.ft = Some(self.fetch_data());
                true
            }
        }
    }
}
