pub trait Model<I, O> {
    fn fit_supervised_dataset(&mut self, _dataset: &Vec<(I, O)>) {
        unimplemented!();
    }

    fn fit_unsupervied_dataset(&mut self, _dataset: &Vec<I>) {
        unimplemented!();
    }

    fn predict(&self, data: &I) -> O;
}
